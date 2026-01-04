use std::time::Instant;
use tantivy::{Index, collector::TopDocs, query::QueryParser};
use regex::RegexBuilder;

use crate::config::SearchConfig;
use crate::error::Result;
use crate::index::schema::SchemaFields;
use super::results::{SearchResult, SearchHit, MatchType};

/// Search engine for querying the index
pub struct Searcher {
    config: SearchConfig,
    index: Index,
    fields: SchemaFields,
}

impl Searcher {
    /// Create a new searcher for an index
    pub fn new(config: SearchConfig, index: Index) -> Self {
        let schema = index.schema();
        let fields = SchemaFields::new(&schema);

        Self {
            config,
            index,
            fields,
        }
    }

    /// Search the index with a query string (literal text matching like grep)
    pub fn search(&self, query: &str, limit: Option<usize>) -> Result<SearchResult> {
        let start = Instant::now();
        let limit = limit.unwrap_or(self.config.default_limit).min(self.config.max_limit);

        // Get a reader
        let reader = self.index.reader()?;
        let searcher = reader.searcher();

        // Build query parser for content field
        let query_parser = QueryParser::for_index(&self.index, vec![self.fields.content]);

        // Extract alphanumeric words for Tantivy query (it can't search special chars)
        // Then we'll post-filter for exact literal match
        let search_terms: Vec<&str> = query
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|s| !s.is_empty())
            .collect();

        // If no searchable terms, return empty
        if search_terms.is_empty() {
            return Ok(SearchResult {
                total: 0,
                hits: vec![],
                query_time_ms: start.elapsed().as_millis() as u64,
                text_hits: 0,
                semantic_hits: 0,
            });
        }

        // Search for the extracted terms
        let tantivy_query_str = search_terms.join(" ");
        let (tantivy_query, _errors) = query_parser.parse_query_lenient(&tantivy_query_str);

        // Fetch more results since we'll filter them down
        let fetch_limit = limit * 10;
        let top_docs = searcher.search(&tantivy_query, &TopDocs::with_limit(fetch_limit))?;

        // Build results
        let mut hits = Vec::with_capacity(top_docs.len());
        let max_score = top_docs.first().map(|(score, _)| *score).unwrap_or(1.0);

        // Case-insensitive literal matching (like grep -i)
        let query_lower = query.to_lowercase();

        for (score, doc_address) in top_docs {
            // Stop if we have enough results
            if hits.len() >= limit {
                break;
            }

            let doc = searcher.doc(doc_address)?;

            // Extract fields
            let path = extract_text(&doc, self.fields.path).unwrap_or_default();
            let doc_id = extract_text(&doc, self.fields.doc_id).unwrap_or_default();
            let content = extract_text(&doc, self.fields.content).unwrap_or_default();
            let line_start = extract_u64(&doc, self.fields.line_start).unwrap_or(1);
            let chunk_id = extract_text(&doc, self.fields.chunk_id).unwrap_or_default();

            // LITERAL GREP-LIKE FILTER: Only include if content contains exact query string
            if !content.to_lowercase().contains(&query_lower) {
                continue;
            }

            // Normalize score to 0-1 range
            let normalized_score = if max_score > 0.0 { score / max_score } else { 0.0 };

            // Create snippet showing lines that match the query
            let (snippet, match_line_offset, snippet_line_count) = create_relevant_snippet(&content, query, 10);

            // Adjust line numbers to reflect where the match actually is
            let actual_line_start = line_start + match_line_offset as u64;
            let actual_line_end = actual_line_start + snippet_line_count.saturating_sub(1) as u64;

            hits.push(SearchHit {
                path,
                line_start: actual_line_start,
                line_end: actual_line_end,
                snippet,
                score: normalized_score,
                is_chunk: !chunk_id.is_empty(),
                doc_id,
                match_type: MatchType::Text,
            });
        }

        let query_time_ms = start.elapsed().as_millis() as u64;
        let text_hits = hits.len();

        Ok(SearchResult {
            total: hits.len(),
            hits,
            query_time_ms,
            text_hits,
            semantic_hits: 0,
        })
    }

    /// Search with filters
    pub fn search_filtered(
        &self,
        query: &str,
        limit: Option<usize>,
        filters: SearchFilters,
        use_regex: bool,
    ) -> Result<SearchResult> {
        // Use regex search if requested
        let mut result = if use_regex {
            self.search_regex(query, Some(limit.unwrap_or(self.config.max_limit) * 2))?
        } else {
            self.search(query, Some(limit.unwrap_or(self.config.max_limit) * 2))?
        };

        // Apply filters
        if let Some(ref extensions) = filters.extensions {
            result.hits.retain(|hit| {
                if let Some(ext) = std::path::Path::new(&hit.path).extension() {
                    extensions.iter().any(|e| e.eq_ignore_ascii_case(&ext.to_string_lossy()))
                } else {
                    false
                }
            });
        }

        if let Some(ref paths) = filters.paths {
            result.hits.retain(|hit| {
                paths.iter().any(|p| hit.path.starts_with(p) || hit.path.contains(p))
            });
        }

        // Re-limit
        let limit = limit.unwrap_or(self.config.default_limit).min(self.config.max_limit);
        result.hits.truncate(limit);
        result.total = result.hits.len();

        Ok(result)
    }

    /// Search the index with a regex pattern
    pub fn search_regex(&self, pattern: &str, limit: Option<usize>) -> Result<SearchResult> {
        let start = Instant::now();
        let limit = limit.unwrap_or(self.config.default_limit).min(self.config.max_limit);

        // Compile regex (case-insensitive by default, like grep -i)
        let regex = match RegexBuilder::new(pattern)
            .case_insensitive(true)
            .build() {
            Ok(r) => r,
            Err(e) => {
                return Err(crate::error::YgrepError::Search(
                    format!("Invalid regex pattern: {}", e)
                ));
            }
        };

        // Get a reader
        let reader = self.index.reader()?;
        let searcher = reader.searcher();

        // Build query parser for content field
        let query_parser = QueryParser::for_index(&self.index, vec![self.fields.content]);

        // Extract alphanumeric words from the regex pattern for Tantivy pre-filter
        // This is a rough heuristic - we extract literal parts from the regex
        let search_terms: Vec<&str> = pattern
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|s| !s.is_empty() && s.len() > 1)  // Skip single chars (likely regex syntax)
            .collect();

        // If we have searchable terms, use Tantivy to narrow down candidates
        let candidates: Vec<_> = if !search_terms.is_empty() {
            let tantivy_query_str = search_terms.join(" ");
            let (tantivy_query, _errors) = query_parser.parse_query_lenient(&tantivy_query_str);

            // Fetch many candidates since regex might be selective
            let fetch_limit = limit * 20;
            searcher.search(&tantivy_query, &TopDocs::with_limit(fetch_limit))?
        } else {
            // No good search terms - scan all documents
            // This is slow but necessary for patterns like "^#" or ".*"
            let all_query = tantivy::query::AllQuery;
            let fetch_limit = limit * 50;
            searcher.search(&all_query, &TopDocs::with_limit(fetch_limit))?
        };

        // Build results by applying regex filter
        let mut hits = Vec::with_capacity(candidates.len());
        let max_score = candidates.first().map(|(score, _)| *score).unwrap_or(1.0);

        for (score, doc_address) in candidates {
            // Stop if we have enough results
            if hits.len() >= limit {
                break;
            }

            let doc = searcher.doc(doc_address)?;

            // Extract fields
            let path = extract_text(&doc, self.fields.path).unwrap_or_default();
            let doc_id = extract_text(&doc, self.fields.doc_id).unwrap_or_default();
            let content = extract_text(&doc, self.fields.content).unwrap_or_default();
            let line_start = extract_u64(&doc, self.fields.line_start).unwrap_or(1);
            let chunk_id = extract_text(&doc, self.fields.chunk_id).unwrap_or_default();

            // REGEX FILTER: Only include if content matches the regex
            if !regex.is_match(&content) {
                continue;
            }

            // Normalize score to 0-1 range
            let normalized_score = if max_score > 0.0 { score / max_score } else { 0.0 };

            // Create snippet showing lines that match the regex
            let (snippet, match_line_offset, snippet_line_count) = create_regex_snippet(&content, &regex, 10);

            // Adjust line numbers to reflect where the match actually is
            let actual_line_start = line_start + match_line_offset as u64;
            let actual_line_end = actual_line_start + snippet_line_count.saturating_sub(1) as u64;

            hits.push(SearchHit {
                path,
                line_start: actual_line_start,
                line_end: actual_line_end,
                snippet,
                score: normalized_score,
                is_chunk: !chunk_id.is_empty(),
                doc_id,
                match_type: MatchType::Text,
            });
        }

        let query_time_ms = start.elapsed().as_millis() as u64;
        let text_hits = hits.len();

        Ok(SearchResult {
            total: hits.len(),
            hits,
            query_time_ms,
            text_hits,
            semantic_hits: 0,
        })
    }
}

/// Filters for search
#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    /// Filter by file extensions (e.g., ["rs", "ts"])
    pub extensions: Option<Vec<String>>,
    /// Filter by path patterns
    pub paths: Option<Vec<String>>,
}

/// Extract text value from a document
fn extract_text(doc: &tantivy::TantivyDocument, field: tantivy::schema::Field) -> Option<String> {
    doc.get_first(field).and_then(|v| {
        if let tantivy::schema::OwnedValue::Str(s) = v {
            Some(s.to_string())
        } else {
            None
        }
    })
}

/// Extract u64 value from a document
fn extract_u64(doc: &tantivy::TantivyDocument, field: tantivy::schema::Field) -> Option<u64> {
    doc.get_first(field).and_then(|v| {
        if let tantivy::schema::OwnedValue::U64(n) = v {
            Some(*n)
        } else {
            None
        }
    })
}

/// Create a snippet showing lines relevant to the query
/// Returns (snippet, line_offset_from_start, line_count)
fn create_relevant_snippet(content: &str, query: &str, max_lines: usize) -> (String, usize, usize) {
    let lines: Vec<&str> = content.lines().collect();
    let query_lower = query.to_lowercase();
    let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

    // Find lines that contain any query term
    let mut matching_indices: Vec<usize> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let line_lower = line.to_lowercase();
        if query_terms.iter().any(|term| line_lower.contains(term)) {
            matching_indices.push(i);
        }
    }

    if matching_indices.is_empty() {
        // No direct matches, return first lines
        let snippet = lines.iter().take(max_lines).copied().collect::<Vec<_>>().join("\n");
        let line_count = snippet.lines().count();
        return (snippet, 0, line_count);
    }

    // Get context around the first match
    let first_match = matching_indices[0];
    let context_before = 2;
    let context_after = max_lines.saturating_sub(context_before + 1);

    let start = first_match.saturating_sub(context_before);
    let end = (first_match + context_after + 1).min(lines.len());

    let snippet = lines[start..end].join("\n");
    let line_count = end - start;
    (snippet, start, line_count)
}

/// Create a snippet showing lines relevant to a regex match
/// Returns (snippet, line_offset_from_start, line_count)
fn create_regex_snippet(content: &str, regex: &regex::Regex, max_lines: usize) -> (String, usize, usize) {
    let lines: Vec<&str> = content.lines().collect();

    // Find lines that match the regex
    let mut matching_indices: Vec<usize> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if regex.is_match(line) {
            matching_indices.push(i);
        }
    }

    if matching_indices.is_empty() {
        // No direct line matches, but document matched - return first lines
        let snippet = lines.iter().take(max_lines).copied().collect::<Vec<_>>().join("\n");
        let line_count = snippet.lines().count();
        return (snippet, 0, line_count);
    }

    // Get context around the first match
    let first_match = matching_indices[0];
    let context_before = 2;
    let context_after = max_lines.saturating_sub(context_before + 1);

    let start = first_match.saturating_sub(context_before);
    let end = (first_match + context_after + 1).min(lines.len());

    let snippet = lines[start..end].join("\n");
    let line_count = end - start;
    (snippet, start, line_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::schema::{build_document_schema, register_tokenizers};
    use tantivy::doc;
    use tempfile::tempdir;

    #[test]
    fn test_basic_search() -> Result<()> {
        let temp_dir = tempdir().unwrap();
        let index_path = temp_dir.path();

        // Create index with schema
        let schema = build_document_schema();
        let index = Index::create_in_dir(index_path, schema.clone())?;

        // Register custom code tokenizer
        register_tokenizers(index.tokenizers());

        let fields = SchemaFields::new(&schema);

        // Add a test document
        let mut writer = index.writer(50_000_000)?;
        writer.add_document(doc!(
            fields.doc_id => "test1",
            fields.path => "src/main.rs",
            fields.workspace => "/test",
            fields.content => "fn main() { println!(\"Hello, world!\"); }",
            fields.mtime => 0u64,
            fields.size => 100u64,
            fields.extension => "rs",
            fields.line_start => 1u64,
            fields.line_end => 1u64,
            fields.chunk_id => "",
            fields.parent_doc => ""
        ))?;
        writer.commit()?;

        // Search
        let config = SearchConfig::default();
        let searcher = Searcher::new(config, index);
        let result = searcher.search("hello", None)?;

        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].path, "src/main.rs");

        Ok(())
    }
}

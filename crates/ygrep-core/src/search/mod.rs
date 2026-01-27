#[cfg(feature = "embeddings")]
mod hybrid;
mod results;
mod searcher;

#[cfg(feature = "embeddings")]
pub use hybrid::HybridSearcher;
pub use results::{MatchType, SearchHit, SearchResult};
pub use searcher::{SearchFilters, Searcher};

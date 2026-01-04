//! Shared utility functions

/// Simple glob matching for ignore patterns.
///
/// Supports the following patterns:
/// - `**/dir/**` - match directory anywhere in path
/// - `**/*.ext` - match extension anywhere
/// - `**/something` - match at end of path
/// - `something/**` - match at start of path
/// - `*.ext` - match extension
/// - exact match or path component match
pub(crate) fn glob_match(pattern: &str, path: &str) -> bool {
    // Handle **/dir/** patterns (match dir anywhere in path)
    if let Some(rest) = pattern.strip_prefix("**/") {
        if let Some(dir_name) = rest.strip_suffix("/**") {
            // Check if this directory name appears as a complete path component
            return path.contains(&format!("/{}/", dir_name))
                || path.starts_with(&format!("{}/", dir_name))
                || path.ends_with(&format!("/{}", dir_name));
        }
    }

    // Handle **/*.ext patterns (match extension anywhere)
    if let Some(ext) = pattern.strip_prefix("**/*.") {
        return path.ends_with(&format!(".{}", ext));
    }

    // Handle **/something patterns (match at end)
    if let Some(suffix) = pattern.strip_prefix("**/") {
        return path.ends_with(suffix) || path.ends_with(&format!("/{}", suffix));
    }

    // Handle something/** patterns (match at start)
    if let Some(prefix) = pattern.strip_suffix("/**") {
        return path.starts_with(prefix) || path.contains(&format!("/{}", prefix));
    }

    // Handle simple * patterns (*.ext)
    if let Some(ext) = pattern.strip_prefix("*.") {
        return path.ends_with(&format!(".{}", ext));
    }

    // Exact match or path component match
    path == pattern
        || path.ends_with(&format!("/{}", pattern))
        || path.contains(&format!("/{}/", pattern))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_match() {
        // **/dir/** patterns
        assert!(glob_match("**/node_modules/**", "foo/node_modules/bar/baz.js"));
        assert!(glob_match("**/.git/**", ".git/config"));

        // **/*.ext patterns
        assert!(glob_match("**/*.log", "foo/bar/debug.log"));
        assert!(!glob_match("**/*.log", "foo/bar/debug.txt"));

        // *.ext patterns
        assert!(glob_match("*.log", "debug.log"));
        assert!(!glob_match("*.log", "debug.txt"));

        // **/something patterns
        assert!(glob_match("**/config", "foo/bar/config"));
        assert!(glob_match("**/config", "config"));

        // something/** patterns
        assert!(glob_match("build/**", "build/output/file.o"));
    }
}

#[deny(missing_docs)]
use std::{
    cmp::Ordering,
    collections::{btree_map, BTreeMap},
    ops::Index,
};

use crate::nfa::{CharacterClass, NFA};

pub mod nfa;

/// The metadata that we will store in the acceptance points of the NFA
#[derive(Clone)]
struct Metadata {
    statics: u32,
    dynamics: u32,
    stars: u32,
    param_names: Vec<String>,
}

impl Metadata {
    /// Create a new Metadata object
    pub fn new() -> Self {
        Self {
            statics: 0,
            dynamics: 0,
            stars: 0,
            param_names: Vec::new(),
        }
    }
}

impl Ord for Metadata {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.statics > other.statics {
            Ordering::Greater
        } else if self.statics < other.statics {
            Ordering::Less
        } else if self.dynamics > other.dynamics {
            Ordering::Greater
        } else if self.dynamics < other.dynamics {
            Ordering::Less
        } else if self.stars > other.stars {
            Ordering::Greater
        } else if self.stars < other.stars {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    }
}

impl PartialOrd for Metadata {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Metadata {
    fn eq(&self, other: &Self) -> bool {
        self.statics == other.statics
            && self.dynamics == other.dynamics
            && self.stars == other.stars
    }
}

impl Eq for Metadata {}

/// A data structure that holds a bunch of parameters that map from one key to a value
#[derive(PartialEq, Clone, Debug, Default)]
pub struct Params {
    map: BTreeMap<String, String>,
}

impl Params {
    /// Create a new parameter set
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    /// Insert a new key and value into the parameter set
    pub fn insert(&mut self, key: String, value: String) {
        self.map.insert(key, value);
    }

    /// Find and return the string relating to a particular key, or `None` otherwise
    pub fn find(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(|s| &s[..])
    }

    /// Iterate over parameters in the set
    pub fn iter(&self) -> Iter<'_> {
        Iter(self.map.iter())
    }
}

impl Index<&str> for Params {
    type Output = String;
    fn index(&self, index: &str) -> &String {
        match self.map.get(index) {
            None => panic!(format!("params[{}] did not exist", index)),
            Some(s) => s,
        }
    }
}

impl<'a> IntoIterator for &'a Params {
    type IntoIter = Iter<'a>;
    type Item = (&'a str, &'a str);

    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

/// An interator over a parameter set
pub struct Iter<'a>(btree_map::Iter<'a, String, String>);

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a str, &'a str);

    #[inline]
    fn next(&mut self) -> Option<(&'a str, &'a str)> {
        self.0.next().map(|(k, v)| (&**k, &**v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

/// A structure that indicates that the router found a match.
/// Comprises of the metadata to return and the parameters set.
pub struct Match<T> {
    pub handler: T,
    pub params: Params,
}

impl<T> Match<T> {
    /// Create a new Match
    pub fn new(handler: T, params: Params) -> Self {
        Self { handler, params }
    }
}

/// A data structure that can recognize and return arbitrary routes with flexible parameter names.
#[derive(Clone)]
pub struct Router<T> {
    nfa: NFA<Metadata>,
    handlers: BTreeMap<usize, T>,
}

impl<T> Router<T> {
    /// Create a new Router.
    pub fn new() -> Self {
        Self {
            nfa: NFA::new(),
            handlers: BTreeMap::new(),
        }
    }

    /// Add a `route` to the Router, with a value to return if the route is recognized.
    pub fn add(&mut self, mut route: &str, dest: T) {
        if !route.is_empty() && route.as_bytes()[0] == b'/' {
            route = &route[1..];
        }

        let nfa = &mut self.nfa;
        let mut state = 0;
        let mut metadata = Metadata::new();
        let segments: Vec<_> = route.split('/').collect();

        for (i, segment) in segments.iter().enumerate() {
            let mut chars = segment.chars();
            let mut is_static = true;
            while let Some(char) = chars.next() {
                if char == '{' {
                    let param_name: String = chars.by_ref().take_while(|c| *c != '}').collect();
                    let (param_name, pattern) =
                        param_name.split_at(param_name.find(':').unwrap_or(param_name.len()));
                    is_static = false;
                    metadata.param_names.push(param_name.into());
                    match pattern.as_bytes().get(1) {
                        // There's no `:`, so its just a parameter
                        None => {
                            metadata.dynamics += 1;
                            state = nfa.put(state, CharacterClass::invalid_char('/'));
                            // Links to itself for repeats
                            nfa.put_state(state, state);
                            nfa.start_capture(state);
                            nfa.end_capture(state);
                        }
                        // * indicates a greedy glob
                        Some(b'*') => {
                            metadata.stars += 1;
                            state = nfa.put(state, CharacterClass::any());
                            // Links to itself for repeats
                            nfa.put_state(state, state);
                            nfa.start_capture(state);
                            nfa.end_capture(state);
                        }
                        // ! indicates that we go until one of the following characters is met
                        Some(b'!') => {
                            metadata.dynamics += 1;
                            let mut pattern = pattern[2..].to_owned();
                            pattern.push('/');
                            state = nfa.put(state, CharacterClass::invalid(&pattern));
                            nfa.put_state(state, state);
                            nfa.start_capture(state);
                            nfa.end_capture(state);
                        }
                        // Otherwise we go whilst we have one of the following characters
                        Some(_) => {
                            metadata.dynamics += 1;
                            let state = nfa.put(state, CharacterClass::valid(&pattern[1..]));
                            nfa.put_state(state, state);
                            nfa.start_capture(state);
                            nfa.end_capture(state);
                        }
                    }
                } else {
                    state = nfa.put(state, CharacterClass::valid_char(char));
                }
            }

            // We've finished this segment, so we need to cap it off with a '/'
            if i != segments.len() - 1 {
                // Create a new state
                state = nfa.put(state, CharacterClass::valid_char('/'));
            }
            if is_static {
                metadata.statics += 1;
            }
        }

        nfa.acceptance(state);
        nfa.metadata(state, metadata);
        self.handlers.insert(state, dest);
    }

    /// Attempt to find a route defined by `path` in the Router
    pub fn recognize(&self, mut path: &str) -> Result<Match<&T>, String> {
        if !path.is_empty() && path.as_bytes()[0] == b'/' {
            path = &path[1..];
        }

        let nfa = &self.nfa;
        let result = nfa.process(path, |index| nfa.get(index).metadata.as_ref().unwrap());

        match result {
            Ok(nfa_match) => {
                let mut map = Params::new();
                let state = &nfa.get(nfa_match.state);
                let metadata = state.metadata.as_ref().unwrap();
                let param_names = metadata.param_names.clone();

                for (i, capture) in nfa_match.captures.iter().enumerate() {
                    if !param_names[i].is_empty() {
                        map.insert(param_names[i].to_string(), capture.to_string());
                    }
                }

                let handler = self.handlers.get(&nfa_match.state).unwrap();
                Ok(Match::new(handler, map))
            }
            Err(err) => Err(err.to_string()),
        }
    }
}

impl<T> Default for Router<T> {
    fn default() -> Self {
        Self::new()
    }
}

fn process_static_segment<T>(segment: &str, nfa: &mut NFA<T>, mut state: usize) -> usize {
    for char in segment.chars() {
        state = nfa.put(state, CharacterClass::valid_char(char));
    }

    state
}

fn process_dynamic_segment<T>(nfa: &mut NFA<T>, mut state: usize) -> usize {
    state = nfa.put(state, CharacterClass::invalid_char('/'));
    nfa.put_state(state, state);
    nfa.start_capture(state);
    nfa.end_capture(state);

    state
}

fn process_star_state<T>(nfa: &mut NFA<T>, mut state: usize) -> usize {
    state = nfa.put(state, CharacterClass::any());
    nfa.put_state(state, state);
    nfa.start_capture(state);
    nfa.end_capture(state);

    state
}

#[cfg(test)]
mod tests {
    use super::{Params, Router};

    #[test]
    fn basic_router() {
        let mut router = Router::new();

        router.add("/thomas", "Thomas".to_string());
        router.add("/tom", "Tom".to_string());
        router.add("/wycats", "Yehuda".to_string());

        let m = router.recognize("/thomas").unwrap();

        assert_eq!(*m.handler, "Thomas".to_string());
        assert_eq!(m.params, Params::new());
    }

    #[test]
    fn root_router() {
        let mut router = Router::new();
        router.add("/", 10);
        assert_eq!(*router.recognize("/").unwrap().handler, 10)
    }

    #[test]
    fn empty_path() {
        let mut router = Router::new();
        router.add("/", 12);
        assert_eq!(*router.recognize("").unwrap().handler, 12)
    }

    #[test]
    fn empty_route() {
        let mut router = Router::new();
        router.add("", 12);
        assert_eq!(*router.recognize("/").unwrap().handler, 12)
    }

    #[test]
    fn ambiguous_router() {
        let mut router = Router::new();

        router.add("/posts/new", "new".to_string());
        router.add("/posts/{id}", "id".to_string());

        let id = router.recognize("/posts/1").unwrap();

        assert_eq!(*id.handler, "id".to_string());
        assert_eq!(id.params, params("id", "1"));

        let new = router.recognize("/posts/new").unwrap();
        assert_eq!(*new.handler, "new".to_string());
        assert_eq!(new.params, Params::new());
    }

    #[test]
    fn ambiguous_router_b() {
        let mut router = Router::new();

        router.add("/posts/{id}", "id".to_string());
        router.add("/posts/new", "new".to_string());

        let id = router.recognize("/posts/1").unwrap();

        assert_eq!(*id.handler, "id".to_string());
        assert_eq!(id.params, params("id", "1"));

        let new = router.recognize("/posts/new").unwrap();
        assert_eq!(*new.handler, "new".to_string());
        assert_eq!(new.params, Params::new());
    }

    #[test]
    fn multiple_params() {
        let mut router = Router::new();

        router.add("/posts/{post_id}/comments/{id}", "comment".to_string());
        router.add("/posts/{post_id}/comments", "comments".to_string());

        let com = router.recognize("/posts/12/comments/100").unwrap();
        let coms = router.recognize("/posts/12/comments").unwrap();

        assert_eq!(*com.handler, "comment".to_string());
        assert_eq!(com.params, two_params("post_id", "12", "id", "100"));

        assert_eq!(*coms.handler, "comments".to_string());
        assert_eq!(coms.params, params("post_id", "12"));
        assert_eq!(coms.params["post_id"], "12".to_string());
    }

    #[test]
    fn star() {
        let mut router = Router::new();

        router.add("{foo:*}", "test".to_string());
        router.add("/bar/{foo:*}", "test2".to_string());

        let m = router.recognize("/test").unwrap();
        assert_eq!(*m.handler, "test".to_string());
        assert_eq!(m.params, params("foo", "test"));

        let m = router.recognize("/foo/bar").unwrap();
        assert_eq!(*m.handler, "test".to_string());
        assert_eq!(m.params, params("foo", "foo/bar"));

        let m = router.recognize("/bar/foo").unwrap();
        assert_eq!(*m.handler, "test2".to_string());
        assert_eq!(m.params, params("foo", "foo"));
    }

    #[test]
    fn star_colon() {
        let mut router = Router::new();

        router.add("/a/{b:*}", "ab".to_string());
        router.add("/a/{b:*}/c", "abc".to_string());
        router.add("/a/{b:*}/c/{d}", "abcd".to_string());

        let m = router.recognize("/a/foo").unwrap();
        assert_eq!(*m.handler, "ab".to_string());
        assert_eq!(m.params, params("b", "foo"));

        let m = router.recognize("/a/foo/bar").unwrap();
        assert_eq!(*m.handler, "ab".to_string());
        assert_eq!(m.params, params("b", "foo/bar"));

        let m = router.recognize("/a/foo/c").unwrap();
        assert_eq!(*m.handler, "abc".to_string());
        assert_eq!(m.params, params("b", "foo"));

        let m = router.recognize("/a/foo/bar/c").unwrap();
        assert_eq!(*m.handler, "abc".to_string());
        assert_eq!(m.params, params("b", "foo/bar"));

        let m = router.recognize("/a/foo/c/baz").unwrap();
        assert_eq!(*m.handler, "abcd".to_string());
        assert_eq!(m.params, two_params("b", "foo", "d", "baz"));

        let m = router.recognize("/a/foo/bar/c/baz").unwrap();
        assert_eq!(*m.handler, "abcd".to_string());
        assert_eq!(m.params, two_params("b", "foo/bar", "d", "baz"));

        let m = router.recognize("/a/foo/bar/c/baz/bay").unwrap();
        assert_eq!(*m.handler, "ab".to_string());
        assert_eq!(m.params, params("b", "foo/bar/c/baz/bay"));
    }

    #[test]
    fn unnamed_parameters() {
        let mut router = Router::new();

        router.add("/foo/{}/bar", "test".to_string());
        router.add("/foo/{bar}/{:*}", "test2".to_string());

        let m = router.recognize("/foo/test/bar").unwrap();
        assert_eq!(*m.handler, "test");
        assert_eq!(m.params, Params::new());

        let m = router.recognize("/foo/test/blah").unwrap();
        assert_eq!(*m.handler, "test2");
        assert_eq!(m.params, params("bar", "test"));
    }

    #[test]
    fn extensions_and_prefixes() {
        let mut router = Router::new();

        router.add("/foo/b{}r", 0);

        assert!(router.recognize("/foo/bar").is_ok());
        assert!(router.recognize("/foo/baaaar").is_ok());
        assert!(router.recognize("/foo/caaaar").is_err());
    }

    fn params(key: &str, val: &str) -> Params {
        let mut map = Params::new();
        map.insert(key.to_string(), val.to_string());
        map
    }

    fn two_params(k1: &str, v1: &str, k2: &str, v2: &str) -> Params {
        let mut map = Params::new();
        map.insert(k1.to_string(), v1.to_string());
        map.insert(k2.to_string(), v2.to_string());
        map
    }
}

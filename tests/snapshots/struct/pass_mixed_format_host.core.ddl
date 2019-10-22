//! Test that a struct with a host type field produces a warning.

struct Test {
    format : primitive U32Be,
    host : !,
}

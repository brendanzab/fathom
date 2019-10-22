//! Operational semantics of the data description language.

use codespan::Span;

use crate::core::{Term, Value};
use crate::ieee754;

/// Evaluate a term into a semantic value.
pub fn eval(term: &Term) -> Value {
    match term {
        Term::Item(_, label) => Value::Item(label.clone()), // TODO: Evaluate to value in environment
        Term::Ann(term, _) => eval(term),
        Term::Universe(_, universe) => Value::Universe(*universe),
        Term::Primitive(_, name) => Value::Primitive(name.clone()),
        Term::IntConst(_, value) => Value::IntConst(value.clone()),
        Term::F32Const(_, value) => Value::F32Const(*value),
        Term::F64Const(_, value) => Value::F64Const(*value),
        Term::Error(_) => Value::Error,
    }
}

/// Read a value back into the term syntax.
pub fn readback(value: &Value) -> Term {
    match value {
        Value::Item(label) => Term::Item(Span::initial(), label.clone()),
        Value::Universe(universe) => Term::Universe(Span::initial(), *universe),
        Value::Primitive(name) => Term::Primitive(Span::initial(), name.clone()),
        Value::IntConst(value) => Term::IntConst(Span::initial(), value.clone()),
        Value::F32Const(value) => Term::F32Const(Span::initial(), *value),
        Value::F64Const(value) => Term::F64Const(Span::initial(), *value),
        Value::Error => Term::Error(Span::initial()),
    }
}

pub fn equal(val1: &Value, val2: &Value) -> bool {
    match (val1, val2) {
        (Value::Item(label0), Value::Item(label1)) => label0 == label1,
        (Value::Universe(universe0), Value::Universe(universe1)) => universe0 == universe1,
        (Value::Primitive(name0), Value::Primitive(name1)) => name0 == name1,
        (Value::IntConst(value0), Value::IntConst(value1)) => value0 == value1,
        (Value::F32Const(value0), Value::F32Const(value1)) => ieee754::logical_eq(*value0, *value1),
        (Value::F64Const(value0), Value::F64Const(value1)) => ieee754::logical_eq(*value0, *value1),
        // Errors are always treated as equal
        (Value::Error, _) | (_, Value::Error) => true,
        // Anything else is not equal!
        (_, _) => false,
    }
}

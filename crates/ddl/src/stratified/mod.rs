//! The core language, syntactically stratified into the two levels of types and
//! terms, with kind annotations erased.

use codespan::{ByteIndex, FileId, Span};
use num_bigint::BigInt;
use std::sync::Arc;

// TODO: move this!
pub use crate::core::Label;

pub mod compile;

/// A module of items.
#[derive(Debug, Clone)]
pub struct Module {
    /// Doc comment.
    pub doc: Arc<[String]>,
    /// The file in which this module was defined.
    pub file_id: FileId,
    /// The items in this module.
    pub items: Vec<Item>,
}

/// Items in a module.
#[derive(Debug, Clone)]
pub enum Item {
    /// Type alias definitions
    TypeAlias(TypeAlias),
    /// Struct definitions.
    StructType(StructType),
    /// Term alias definitions
    TermAlias(TermAlias),
}

impl Item {
    pub fn span(&self) -> Span {
        match self {
            Item::TypeAlias(ty_alias) => ty_alias.span,
            Item::StructType(struct_ty) => struct_ty.span,
            Item::TermAlias(term_alias) => term_alias.span,
        }
    }
}

/// A type alias definition.
#[derive(Debug, Clone)]
pub struct TypeAlias {
    /// The full span of this definition.
    pub span: Span,
    /// Doc comment.
    pub doc: Arc<[String]>,
    /// Name of this definition.
    pub name: Label,
    /// The term that is aliased.
    pub ty: Type,
}

/// A type alias definition.
#[derive(Debug, Clone)]
pub struct TermAlias {
    /// The full span of this definition.
    pub span: Span,
    /// Doc comment.
    pub doc: Arc<[String]>,
    /// Name of this definition.
    pub name: Label,
    /// The term that is aliased.
    pub term: Term,
}

/// A struct type definition.
#[derive(Debug, Clone)]
pub struct StructType {
    /// The full span of this definition.
    pub span: Span,
    /// Doc comment.
    pub doc: Arc<[String]>,
    /// Name of this definition.
    pub name: Label,
    /// Fields in the struct.
    pub fields: Vec<TypeField>,
}

/// A field in a struct type definition.
#[derive(Debug, Clone)]
pub struct TypeField {
    pub doc: Arc<[String]>,
    pub start: ByteIndex,
    pub name: Label,
    pub ty: Type,
}

impl TypeField {
    pub fn span(&self) -> Span {
        Span::new(self.start, self.ty.span().end())
    }
}

/// Types.
#[derive(Debug, Clone)]
pub enum Type {
    /// Item references
    Item(Span, Label),

    /// Unsigned 8-bit integer type.
    U8Type(Span),
    /// Unsigned 16-bit integer type (little endian).
    U16LeType(Span),
    /// Unsigned 16-bit integer type (big endian).
    U16BeType(Span),
    /// Unsigned 32-bit integer type (little endian).
    U32LeType(Span),
    /// Unsigned 32-bit integer type (big endian).
    U32BeType(Span),
    /// Unsigned 64-bit integer type (little endian).
    U64LeType(Span),
    /// Unsigned 64-bit integer type (big endian).
    U64BeType(Span),
    /// Signed, two's complement 8-bit integer type.
    S8Type(Span),
    /// Signed, two's complement 16-bit integer type (little endian).
    S16LeType(Span),
    /// Signed, two's complement 16-bit integer type (big endian).
    S16BeType(Span),
    /// Signed, two's complement 32-bit integer type (little endian).
    S32LeType(Span),
    /// Signed, two's complement 32-bit integer type (big endian).
    S32BeType(Span),
    /// Signed, two's complement 64-bit integer type (little endian).
    S64LeType(Span),
    /// Signed, two's complement 64-bit integer type (big endian).
    S64BeType(Span),
    /// IEEE-754 single-precision floating point number type (little endian).
    F32LeType(Span),
    /// IEEE-754 single-precision floating point number type (big endian).
    F32BeType(Span),
    /// IEEE-754 double-precision floating point number type (little endian).
    F64LeType(Span),
    /// IEEE-754 double-precision floating point number type (big endian).
    F64BeType(Span),

    /// Host boolean type.
    BoolType(Span),
    /// Host integer type.
    IntType(Span),
    /// Host IEEE-754 single-precision floating point type.
    F32Type(Span),
    /// Host IEEE-754 double-precision floating point type.
    F64Type(Span),

    /// Error sentinel.
    Error(Span),
}

impl Type {
    pub fn span(&self) -> Span {
        match self {
            Type::Item(span, _)
            | Type::U8Type(span)
            | Type::U16LeType(span)
            | Type::U16BeType(span)
            | Type::U32LeType(span)
            | Type::U32BeType(span)
            | Type::U64LeType(span)
            | Type::U64BeType(span)
            | Type::S8Type(span)
            | Type::S16LeType(span)
            | Type::S16BeType(span)
            | Type::S32LeType(span)
            | Type::S32BeType(span)
            | Type::S64LeType(span)
            | Type::S64BeType(span)
            | Type::F32LeType(span)
            | Type::F32BeType(span)
            | Type::F64LeType(span)
            | Type::F64BeType(span)
            | Type::BoolType(span)
            | Type::IntType(span)
            | Type::F32Type(span)
            | Type::F64Type(span)
            | Type::Error(span) => *span,
        }
    }
}

/// Terms.
#[derive(Debug, Clone)]
pub enum Term {
    /// Item references
    Item(Span, Label),

    /// Terms annotated with types.
    Ann(Arc<Term>, Arc<Type>),

    /// Host boolean constant.
    BoolConst(Span, bool),
    /// Host integer constants.
    IntConst(Span, BigInt),
    /// Host IEEE-754 single-precision floating point constants.
    F32Const(Span, f32),
    /// Host IEEE-754 double-precision floating point constants.
    F64Const(Span, f64),

    /// Error sentinel.
    Error(Span),
}

impl Term {
    pub fn span(&self) -> Span {
        match self {
            Term::Item(span, _)
            | Term::BoolConst(span, _)
            | Term::IntConst(span, _)
            | Term::F32Const(span, _)
            | Term::F64Const(span, _)
            | Term::Error(span) => *span,
            Term::Ann(term, ty) => Span::merge(term.span(), ty.span()),
        }
    }
}

use codespan::{FileId, Span};
use codespan_reporting::diagnostic::Diagnostic;
use inflector::Inflector;
use num_bigint::BigInt;
use std::collections::HashMap;

use crate::{rust, stratified};

mod diagnostics;

pub fn compile_module(
    module: &stratified::Module,
    report: &mut dyn FnMut(Diagnostic),
) -> rust::Module {
    let mut context = ModuleContext {
        file_id: module.file_id,
        items: HashMap::new(),
    };

    let items = module.items.iter().filter_map(|stratified_item| {
        use std::collections::hash_map::Entry;

        let (label, compiled_item, item) = compile_item(&context, stratified_item, report);
        match context.items.entry(label) {
            Entry::Occupied(entry) => {
                report(diagnostics::bug::item_name_reused(
                    context.file_id,
                    entry.key(),
                    stratified_item.span(),
                    entry.get().span(),
                ));
                None
            }
            Entry::Vacant(entry) => {
                entry.insert(compiled_item);
                item
            }
        }
    });

    rust::Module {
        doc: module.doc.clone(),
        items: items.collect(),
    }
}

#[derive(Debug, Copy, Clone)]
struct CopyTrait;

#[derive(Debug, Clone)]
struct BinaryTrait {
    host_ty: rust::Type,
}

#[derive(Debug, Clone)]
struct Traits {
    copy: Option<CopyTrait>,
    binary: Option<BinaryTrait>,
}

#[derive(Debug, Clone)]
enum CompiledItem {
    Term(Span, String, rust::Type),
    Type(Span, String, Traits),
    Error(Span),
}

impl CompiledItem {
    fn span(&self) -> Span {
        match self {
            CompiledItem::Term(span, _, _)
            | CompiledItem::Type(span, _, _)
            | CompiledItem::Error(span) => *span,
        }
    }
}

struct ModuleContext {
    file_id: FileId,
    items: HashMap<stratified::Label, CompiledItem>,
}

fn compile_item(
    context: &ModuleContext,
    stratified_item: &stratified::Item,
    report: &mut dyn FnMut(Diagnostic),
) -> (stratified::Label, CompiledItem, Option<rust::Item>) {
    match stratified_item {
        stratified::Item::TypeAlias(stratified_ty_alias) => {
            compile_ty_alias(context, stratified_ty_alias, report)
        }
        stratified::Item::StructType(stratified_struct_ty) => {
            compile_struct_ty(context, stratified_struct_ty, report)
        }
        stratified::Item::TermAlias(stratified_ty_alias) => {
            compile_term_alias(context, stratified_ty_alias, report)
        }
    }
}

fn compile_ty_alias(
    context: &ModuleContext,
    stratified_ty_alias: &stratified::TypeAlias,
    report: &mut dyn FnMut(Diagnostic),
) -> (stratified::Label, CompiledItem, Option<rust::Item>) {
    let span = stratified_ty_alias.span;
    match compile_ty(context, &stratified_ty_alias.ty, report) {
        CompiledTerm::Term(term, ty) => unimplemented!(),
        CompiledTerm::Type(ty, traits) => {
            let doc = stratified_ty_alias.doc.clone();
            let name = stratified_ty_alias.name.0.to_pascal_case(); // TODO: name avoidance

            (
                stratified_ty_alias.name.clone(),
                CompiledItem::Type(span, name.clone(), traits),
                Some(rust::Item::TypeAlias(rust::TypeAlias { doc, name, ty })),
            )
        }
        CompiledTerm::Error => (
            stratified_ty_alias.name.clone(),
            CompiledItem::Error(span),
            None,
        ),
    }
}

fn compile_term_alias(
    context: &ModuleContext,
    stratified_term_alias: &stratified::TermAlias,
    report: &mut dyn FnMut(Diagnostic),
) -> (stratified::Label, CompiledItem, Option<rust::Item>) {
    let span = stratified_term_alias.span;
    match compile_term(context, &stratified_term_alias.term, report) {
        CompiledTerm::Term(term, ty) => {
            let doc = stratified_term_alias.doc.clone();
            let name = stratified_term_alias.name.0.to_screaming_snake_case(); // TODO: name avoidance

            (
                stratified_term_alias.name.clone(),
                CompiledItem::Term(span, name.clone(), ty.clone()),
                Some(rust::Item::Const(rust::Const {
                    doc,
                    name,
                    ty,
                    term,
                })),
            )
        }
        CompiledTerm::Type(ty, traits) => unimplemented!(),
        CompiledTerm::Error => (
            stratified_term_alias.name.clone(),
            CompiledItem::Error(span),
            None,
        ),
    }
}

fn compile_struct_ty(
    context: &ModuleContext,
    stratified_struct_ty: &stratified::StructType,
    report: &mut dyn FnMut(Diagnostic),
) -> (stratified::Label, CompiledItem, Option<rust::Item>) {
    const INVALID_TYPE: rust::Type = rust::Type::Rt(rust::RtType::InvalidDataDescription);
    let error = |field: &stratified::TypeField| {
        (
            stratified_struct_ty.name.clone(),
            CompiledItem::Error(field.span()),
            None,
        )
    };

    let mut copy = Some(CopyTrait);
    let mut fields = Vec::with_capacity(stratified_struct_ty.fields.len());

    for field in &stratified_struct_ty.fields {
        let (format_ty, host_ty, field_copy) = match compile_ty(context, &field.ty, report) {
            CompiledTerm::Term(_, _) => {
                // TODO: Bug!
                return error(field);
            }
            CompiledTerm::Type(ty, traits) => match &traits.binary {
                Some(binary) => (ty, binary.host_ty.clone(), traits.copy),
                None => {
                    report(diagnostics::warning::host_type_found_in_field(
                        context.file_id,
                        stratified_struct_ty.span,
                        field.ty.span(),
                    ));
                    return error(field);
                }
            },
            CompiledTerm::Error => (INVALID_TYPE, INVALID_TYPE, None),
        };

        copy = Option::and(copy, field_copy);
        fields.push(rust::TypeField {
            doc: field.doc.clone(),
            name: field.name.0.clone(),
            format_ty,
            host_ty,
        })
    }

    let doc = stratified_struct_ty.doc.clone();
    let name = stratified_struct_ty.name.0.to_pascal_case(); // TODO: name avoidance
    let mut derives = Vec::new();
    if copy.is_some() {
        derives.push("Copy".to_owned());
        derives.push("Clone".to_owned());
    }
    let binary = Some(BinaryTrait {
        host_ty: rust::Type::Var(name.clone()),
    });

    (
        stratified_struct_ty.name.clone(),
        CompiledItem::Type(
            stratified_struct_ty.span,
            name.clone(),
            Traits { copy, binary },
        ),
        Some(rust::Item::Struct(rust::StructType {
            derives,
            doc,
            name,
            fields,
        })),
    )
}

enum CompiledTerm {
    Term(rust::Term, rust::Type),
    Type(rust::Type, Traits),
    Error,
}

fn compile_ty(
    context: &ModuleContext,
    stratified_ty: &stratified::Type,
    report: &mut dyn FnMut(Diagnostic),
) -> CompiledTerm {
    let file_id = context.file_id;

    let host_ty = |ty, copy| CompiledTerm::Type(ty, Traits { copy, binary: None });
    let format_ty = |ty, host_ty| {
        let copy = Some(CopyTrait);
        let binary = Some(BinaryTrait { host_ty });
        CompiledTerm::Type(ty, Traits { copy, binary })
    };

    match stratified_ty {
        stratified::Type::Item(span, label) => match context.items.get(label) {
            Some(CompiledItem::Term(_, _, _)) => unimplemented!(),
            Some(CompiledItem::Type(_, ty_name, traits)) => {
                CompiledTerm::Type(rust::Type::Var(ty_name.clone()), traits.clone())
            }
            Some(CompiledItem::Error(_)) => CompiledTerm::Error,
            None => {
                report(diagnostics::bug::unbound_item(file_id, label, *span));
                CompiledTerm::Error
            }
        },
        stratified::Type::U8Type(_) => format_ty(rust::Type::Rt(rust::RtType::U8), rust::Type::U8),
        stratified::Type::U16LeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::U16Le), rust::Type::U16)
        }
        stratified::Type::U16BeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::U16Be), rust::Type::U16)
        }
        stratified::Type::U32LeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::U32Le), rust::Type::U32)
        }
        stratified::Type::U32BeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::U32Be), rust::Type::U32)
        }
        stratified::Type::U64LeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::U64Le), rust::Type::U64)
        }
        stratified::Type::U64BeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::U64Be), rust::Type::U64)
        }
        stratified::Type::S8Type(_) => format_ty(rust::Type::Rt(rust::RtType::I8), rust::Type::I8),
        stratified::Type::S16LeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::I16Le), rust::Type::I16)
        }
        stratified::Type::S16BeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::I16Be), rust::Type::I16)
        }
        stratified::Type::S32LeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::I32Le), rust::Type::I32)
        }
        stratified::Type::S32BeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::I32Be), rust::Type::I32)
        }
        stratified::Type::S64LeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::I64Le), rust::Type::I64)
        }
        stratified::Type::S64BeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::I64Be), rust::Type::I64)
        }
        stratified::Type::F32LeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::F32Le), rust::Type::F32)
        }
        stratified::Type::F32BeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::F32Be), rust::Type::F32)
        }
        stratified::Type::F64LeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::F64Le), rust::Type::F64)
        }
        stratified::Type::F64BeType(_) => {
            format_ty(rust::Type::Rt(rust::RtType::F64Be), rust::Type::F64)
        }
        stratified::Type::BoolType(_) => host_ty(rust::Type::Bool, Some(CopyTrait)),
        stratified::Type::IntType(span) => {
            report(diagnostics::error::unconstrained_int(file_id, *span));
            host_ty(rust::Type::Rt(rust::RtType::InvalidDataDescription), None)
        }
        stratified::Type::F32Type(_) => host_ty(rust::Type::F32, Some(CopyTrait)),
        stratified::Type::F64Type(_) => host_ty(rust::Type::F64, Some(CopyTrait)),
        stratified::Type::Error(_) => CompiledTerm::Error,
    }
}

fn compile_term(
    context: &ModuleContext,
    stratified_term: &stratified::Term,
    report: &mut dyn FnMut(Diagnostic),
) -> CompiledTerm {
    let file_id = context.file_id;

    match stratified_term {
        stratified::Term::Item(span, label) => match context.items.get(label) {
            Some(CompiledItem::Term(_, name, ty)) => {
                CompiledTerm::Term(rust::Term::Var(name.clone()), ty.clone())
            }
            Some(CompiledItem::Type(_, _, _)) => unimplemented!(),
            Some(CompiledItem::Error(_)) => CompiledTerm::Error,
            None => {
                report(diagnostics::bug::unbound_item(file_id, label, *span));
                CompiledTerm::Error
            }
        },
        stratified::Term::Ann(term, _) => compile_term(context, term, report),
        stratified::Term::BoolConst(_, value) => {
            CompiledTerm::Term(rust::Term::Bool(*value), rust::Type::Bool)
        }
        stratified::Term::IntConst(span, value) => {
            use num_traits::cast::ToPrimitive;

            match value.to_i64() {
                // TODO: don't default to I64.
                Some(value) => CompiledTerm::Term(rust::Term::I64(value), rust::Type::I64),
                None => {
                    report(crate::diagnostics::bug::not_yet_implemented(
                        context.file_id,
                        *span,
                        "non-i64 types",
                    ));
                    CompiledTerm::Error
                }
            }
        }
        stratified::Term::F32Const(_, value) => {
            CompiledTerm::Term(rust::Term::F32(*value), rust::Type::F32)
        }
        stratified::Term::F64Const(_, value) => {
            CompiledTerm::Term(rust::Term::F64(*value), rust::Type::F64)
        }
        stratified::Term::Error(_) => CompiledTerm::Error,
    }
}

#[allow(dead_code)]
fn host_int(min: &BigInt, max: &BigInt) -> Option<rust::Type> {
    use std::{i16, i32, i64, i8, u16, u32, u64, u8};

    match () {
        () if *min >= u8::MIN.into() && *max <= u8::MAX.into() => Some(rust::Type::U8),
        () if *min >= u16::MIN.into() && *max <= u16::MAX.into() => Some(rust::Type::U16),
        () if *min >= u32::MIN.into() && *max <= u32::MAX.into() => Some(rust::Type::U32),
        () if *min >= u64::MIN.into() && *max <= u64::MAX.into() => Some(rust::Type::U64),
        () if *min >= i8::MIN.into() && *max <= i8::MAX.into() => Some(rust::Type::I8),
        () if *min >= i16::MIN.into() && *max <= i16::MAX.into() => Some(rust::Type::I16),
        () if *min >= i32::MIN.into() && *max <= i32::MAX.into() => Some(rust::Type::I32),
        () if *min >= i64::MIN.into() && *max <= i64::MAX.into() => Some(rust::Type::I64),
        () if min > max => None, // Impossible range
        _ => None,               // TODO: use bigint if outside bounds
    }
}

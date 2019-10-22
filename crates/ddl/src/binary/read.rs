use num_bigint::BigInt;
use std::collections::HashMap;

use crate::binary::Term;
use crate::core;

/// Contextual information to be used when parsing items.
pub struct ItemContext<'module> {
    items: HashMap<core::Label, &'module core::Item>,
}

impl<'module> ItemContext<'module> {
    /// Create a new item context.
    pub fn new() -> ItemContext<'module> {
        ItemContext {
            items: HashMap::new(),
        }
    }
}

pub fn read_module_item(
    module: &core::Module,
    name: &str,
    ctxt: &mut ddl_rt::ReadCtxt<'_>,
) -> Result<Term, ddl_rt::ReadError> {
    let mut context = ItemContext::new();

    for item in &module.items {
        match item {
            core::Item::Alias(alias) if alias.name.0 == name => {
                return read_ty(&context, &alias.term, ctxt);
            }
            core::Item::Struct(struct_ty) if struct_ty.name.0 == name => {
                return read_struct_ty(&context, struct_ty, ctxt);
            }
            core::Item::Alias(alias) => {
                context.items.insert(alias.name.clone(), item);
            }
            core::Item::Struct(struct_ty) => {
                context.items.insert(struct_ty.name.clone(), item);
            }
        }
    }

    Err(ddl_rt::ReadError::InvalidDataDescription)
}

pub fn read_struct_ty(
    context: &ItemContext<'_>,
    struct_ty: &core::StructType,
    ctxt: &mut ddl_rt::ReadCtxt<'_>,
) -> Result<Term, ddl_rt::ReadError> {
    let fields = struct_ty
        .fields
        .iter()
        .map(|field| Ok((field.name.0.clone(), read_ty(context, &field.term, ctxt)?)))
        .collect::<Result<_, ddl_rt::ReadError>>()?;

    Ok(Term::Struct(fields))
}

pub fn read_ty(
    context: &ItemContext<'_>,
    term: &core::Term,
    ctxt: &mut ddl_rt::ReadCtxt<'_>,
) -> Result<Term, ddl_rt::ReadError> {
    match term {
        core::Term::Item(_, label) => match context.items.get(label) {
            Some(core::Item::Alias(alias)) => read_ty(&context, &alias.term, ctxt),
            Some(core::Item::Struct(struct_ty)) => read_struct_ty(&context, struct_ty, ctxt),
            None => Err(ddl_rt::ReadError::InvalidDataDescription),
        },
        core::Term::Ann(term, _) => read_ty(context, term, ctxt),
        core::Term::Primitive(_, name) => read_primitive_ty(name, ctxt),
        core::Term::Universe(_, _)
        | core::Term::IntConst(_, _)
        | core::Term::F32Const(_, _)
        | core::Term::F64Const(_, _)
        | core::Term::Error(_) => Err(ddl_rt::ReadError::InvalidDataDescription),
    }
}

pub fn read_primitive_ty(
    name: &str,
    ctxt: &mut ddl_rt::ReadCtxt<'_>,
) -> Result<Term, ddl_rt::ReadError> {
    match name {
        "U8" => Ok(Term::Int(BigInt::from(ctxt.read::<ddl_rt::U8>()?))),
        "U16Le" => Ok(Term::Int(BigInt::from(ctxt.read::<ddl_rt::U16Le>()?))),
        "U16Be" => Ok(Term::Int(BigInt::from(ctxt.read::<ddl_rt::U16Be>()?))),
        "U32Le" => Ok(Term::Int(BigInt::from(ctxt.read::<ddl_rt::U32Le>()?))),
        "U32Be" => Ok(Term::Int(BigInt::from(ctxt.read::<ddl_rt::U32Be>()?))),
        "U64Le" => Ok(Term::Int(BigInt::from(ctxt.read::<ddl_rt::U64Le>()?))),
        "U64Be" => Ok(Term::Int(BigInt::from(ctxt.read::<ddl_rt::U64Be>()?))),
        "S8" => Ok(Term::Int(BigInt::from(ctxt.read::<ddl_rt::I8>()?))),
        "S16Le" => Ok(Term::Int(BigInt::from(ctxt.read::<ddl_rt::I16Le>()?))),
        "S16Be" => Ok(Term::Int(BigInt::from(ctxt.read::<ddl_rt::I16Be>()?))),
        "S32Le" => Ok(Term::Int(BigInt::from(ctxt.read::<ddl_rt::I32Le>()?))),
        "S32Be" => Ok(Term::Int(BigInt::from(ctxt.read::<ddl_rt::I32Be>()?))),
        "S64Le" => Ok(Term::Int(BigInt::from(ctxt.read::<ddl_rt::I64Le>()?))),
        "S64Be" => Ok(Term::Int(BigInt::from(ctxt.read::<ddl_rt::I64Be>()?))),
        "F32Le" => Ok(Term::F32(ctxt.read::<ddl_rt::F32Le>()?)),
        "F32Be" => Ok(Term::F32(ctxt.read::<ddl_rt::F32Be>()?)),
        "F64Le" => Ok(Term::F64(ctxt.read::<ddl_rt::F64Le>()?)),
        "F64Be" => Ok(Term::F64(ctxt.read::<ddl_rt::F64Be>()?)),
        _ => Err(ddl_rt::ReadError::InvalidDataDescription),
    }
}

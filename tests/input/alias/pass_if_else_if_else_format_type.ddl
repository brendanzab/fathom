Test : Format =
    if true {
        if true { F64Be } else { F32Be }
    } else {
        if false { F64Be } else { F32Be }
    };

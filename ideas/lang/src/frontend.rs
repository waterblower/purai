//! Source -> syntax -> checked, constant-string IR. No platform details here.
use crate::Diagnostic;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
enum Kind {
    Ident(String),
    String(String),
    LParen,
    RParen,
    LBrace,
    RBrace,
    Equal,
    Comma,
    Semi,
    Eof,
}

#[derive(Debug, Clone)]
struct Token {
    kind: Kind,
    offset: usize,
}

fn lex(source: &str) -> Result<Vec<Token>, Diagnostic> {
    let mut chars = source.char_indices().peekable();
    let mut tokens = Vec::new();
    while let Some((offset, c)) = chars.next() {
        let kind = match c {
            c if c.is_whitespace() => continue,
            '/' if chars.peek().is_some_and(|(_, c)| *c == '/') => {
                for (_, c) in chars.by_ref() {
                    if c == '\n' {
                        break;
                    }
                }
                continue;
            }
            '(' => Kind::LParen,
            ')' => Kind::RParen,
            '{' => Kind::LBrace,
            '}' => Kind::RBrace,
            '=' => Kind::Equal,
            ',' => Kind::Comma,
            ';' => Kind::Semi,
            '"' => {
                let mut value = String::new();
                let mut closed = false;
                while let Some((at, c)) = chars.next() {
                    match c {
                        '"' => {
                            closed = true;
                            break;
                        }
                        '\\' => {
                            let Some((escape_at, escape)) = chars.next() else {
                                return Err(Diagnostic::new(at, "unfinished string escape"));
                            };
                            value.push(match escape {
                                'n' => '\n',
                                'r' => '\r',
                                't' => '\t',
                                '0' => '\0',
                                '\\' => '\\',
                                '"' => '"',
                                _ => {
                                    return Err(Diagnostic::new(
                                        escape_at,
                                        format!("unsupported escape \\{escape}"),
                                    ));
                                }
                            });
                        }
                        _ => value.push(c),
                    }
                }
                if !closed {
                    return Err(Diagnostic::new(offset, "unterminated string literal"));
                }
                Kind::String(value)
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let mut name = String::from(c);
                while let Some(&(_, c)) = chars.peek() {
                    if !c.is_ascii_alphanumeric() && c != '_' {
                        break;
                    }
                    name.push(c);
                    chars.next();
                }
                Kind::Ident(name)
            }
            _ => {
                return Err(Diagnostic::new(
                    offset,
                    format!("unexpected character {c:?}"),
                ));
            }
        };
        tokens.push(Token { kind, offset });
    }
    tokens.push(Token {
        kind: Kind::Eof,
        offset: source.len(),
    });
    Ok(tokens)
}

enum Expr {
    Literal(String),
    Name(Token),
}
enum Statement {
    Let { name: Token, value: Expr },
    Call { name: Token, args: Vec<Expr> },
}
struct Function {
    name: Token,
    body: Vec<Statement>,
}
struct Parser {
    tokens: Vec<Token>,
    at: usize,
}

impl Parser {
    fn peek(&self) -> &Token {
        &self.tokens[self.at]
    }
    fn take(&mut self) -> Token {
        let token = self.peek().clone();
        if token.kind != Kind::Eof {
            self.at += 1;
        }
        token
    }
    fn expect(&mut self, kind: Kind, description: &str) -> Result<(), Diagnostic> {
        if self.peek().kind != kind {
            return Err(Diagnostic::new(
                self.peek().offset,
                format!("expected {description}"),
            ));
        }
        self.take();
        Ok(())
    }
    fn ident(&mut self) -> Result<Token, Diagnostic> {
        match &self.peek().kind {
            Kind::Ident(name) if name != "fn" && name != "let" => Ok(self.take()),
            _ => Err(Diagnostic::new(self.peek().offset, "expected a name")),
        }
    }
    fn expr(&mut self) -> Result<Expr, Diagnostic> {
        let token = self.take();
        match &token.kind {
            Kind::String(s) => Ok(Expr::Literal(s.clone())),
            Kind::Ident(s) if s != "fn" && s != "let" => Ok(Expr::Name(token)),
            _ => Err(Diagnostic::new(
                token.offset,
                "expected a string literal or binding",
            )),
        }
    }
    fn program(&mut self) -> Result<Vec<Function>, Diagnostic> {
        let mut functions = Vec::new();
        while self.peek().kind != Kind::Eof {
            self.expect(Kind::Ident("fn".into()), "`fn`")?;
            let name = self.ident()?;
            self.expect(Kind::LParen, "`(`")?;
            self.expect(
                Kind::RParen,
                "`)` (function parameters are not implemented yet)",
            )?;
            self.expect(Kind::LBrace, "`{`")?;
            let mut body = Vec::new();
            while self.peek().kind != Kind::RBrace {
                if self.peek().kind == Kind::Eof {
                    return Err(Diagnostic::new(self.peek().offset, "expected `}`"));
                }
                if self.peek().kind == Kind::Semi {
                    self.take();
                    continue;
                }
                if self.peek().kind == Kind::Ident("let".into()) {
                    self.take();
                    let name = self.ident()?;
                    self.expect(Kind::Equal, "`=`")?;
                    body.push(Statement::Let {
                        name,
                        value: self.expr()?,
                    });
                } else {
                    let name = self.ident()?;
                    self.expect(Kind::LParen, "`(`")?;
                    let mut args = Vec::new();
                    if self.peek().kind != Kind::RParen {
                        loop {
                            args.push(self.expr()?);
                            if self.peek().kind != Kind::Comma {
                                break;
                            }
                            self.take();
                        }
                    }
                    self.expect(Kind::RParen, "`)`")?;
                    body.push(Statement::Call { name, args });
                }
            }
            self.take();
            functions.push(Function { name, body });
        }
        Ok(functions)
    }
}

fn name(token: &Token) -> &str {
    match &token.kind {
        Kind::Ident(s) => s,
        _ => unreachable!(),
    }
}

pub enum Op {
    Write(Vec<u8>),
    Call(usize),
}
pub struct CheckedFunction {
    pub ops: Vec<Op>,
}
pub struct Program {
    pub functions: Vec<CheckedFunction>,
    pub main: usize,
}

fn resolve(expr: Expr, locals: &HashMap<String, String>) -> Result<String, Diagnostic> {
    match expr {
        Expr::Literal(s) => Ok(s),
        Expr::Name(token) => locals.get(name(&token)).cloned().ok_or_else(|| {
            Diagnostic::new(
                token.offset,
                format!("unknown string binding `{}`", name(&token)),
            )
        }),
    }
}

pub fn lower(source: &str) -> Result<Program, Diagnostic> {
    let functions = Parser {
        tokens: lex(source)?,
        at: 0,
    }
    .program()?;
    let mut names = HashMap::new();
    for (id, function) in functions.iter().enumerate() {
        let n = name(&function.name);
        if n == "print" || n == "println" {
            return Err(Diagnostic::new(
                function.name.offset,
                format!("`{n}` is a reserved builtin"),
            ));
        }
        if names.insert(n.to_owned(), id).is_some() {
            return Err(Diagnostic::new(
                function.name.offset,
                format!("duplicate function `{n}`"),
            ));
        }
    }
    let main = *names
        .get("main")
        .ok_or_else(|| Diagnostic::new(0, "missing `fn main()`"))?;
    let mut checked = Vec::new();
    for function in functions {
        let mut locals = HashMap::new();
        let mut ops = Vec::new();
        for statement in function.body {
            match statement {
                Statement::Let { name: token, value } => {
                    let value = resolve(value, &locals)?;
                    locals.insert(name(&token).to_owned(), value);
                }
                Statement::Call {
                    name: token,
                    mut args,
                } => {
                    let n = name(&token);
                    if locals.contains_key(n) {
                        return Err(Diagnostic::new(
                            token.offset,
                            format!("`{n}` has type `str`, which is not callable"),
                        ));
                    }
                    if n == "print" || n == "println" {
                        if args.len() != 1 {
                            return Err(Diagnostic::new(
                                token.offset,
                                format!("`{n}` expects one `str` argument"),
                            ));
                        }
                        let mut value = resolve(args.pop().unwrap(), &locals)?.into_bytes();
                        if n == "println" {
                            value.push(b'\n');
                        }
                        ops.push(Op::Write(value));
                    } else {
                        let id = *names.get(n).ok_or_else(|| {
                            Diagnostic::new(token.offset, format!("unknown function `{n}`"))
                        })?;
                        if !args.is_empty() {
                            return Err(Diagnostic::new(
                                token.offset,
                                format!("`{n}` expects no arguments"),
                            ));
                        }
                        ops.push(Op::Call(id));
                    }
                }
            }
        }
        checked.push(CheckedFunction { ops });
    }
    Ok(Program {
        functions: checked,
        main,
    })
}

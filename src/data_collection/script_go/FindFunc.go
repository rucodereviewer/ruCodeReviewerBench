package main

import (
    "fmt"
    "go/ast"
    "go/parser"
    "go/token"
    "os"
    "strconv"
)

func main() {
    if len(os.Args) != 3 {
        fmt.Println("Usage: go run FindFunc.go <filename> <line_number>")
        return
    }

    lineNumber, err := strconv.Atoi(os.Args[1])
    filename := os.Args[2]

    if err != nil {
        fmt.Printf("Invalid line number: %v\n", err)
        return
    }

    fset := token.NewFileSet()

    node, err := parser.ParseFile(fset, filename, nil, 0)
    if err != nil {
        fmt.Printf("Error parsing file: %v\n", err)
        return
    }

    for _, decl := range node.Decls {
        switch d := decl.(type) {
        case *ast.FuncDecl:
            start := fset.Position(d.Pos()).Line
            end := fset.Position(d.End()).Line

            if start <= lineNumber && lineNumber <= end {
                fmt.Printf("%d %d\n", start, end)
            }

        case *ast.GenDecl:
            for _, spec := range d.Specs {
                if typeSpec, ok := spec.(*ast.TypeSpec); ok {
                    if _, ok := typeSpec.Type.(*ast.StructType); ok {
                        start := fset.Position(d.Pos()).Line
                        end := fset.Position(d.End()).Line

                        if start <= lineNumber && lineNumber <= end {
                            fmt.Printf("%d %d\n", start, end)
                        }
                    }
                }
            }
        }
    }
}
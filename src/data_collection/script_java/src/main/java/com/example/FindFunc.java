import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.TreeVisitor;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

public class FindFunc {
    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Usage: java FindFunc <line-to-find> <source-file-path>");
            return;
        }

        int lineNumber = Integer.parseInt(args[0]);
        String filePath = args[1];

        try {
            File file = Paths.get(filePath).toFile();
            ParserConfiguration parserConfiguration = new ParserConfiguration();
            parserConfiguration.setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_17);
            StaticJavaParser.setConfiguration(parserConfiguration);
            
            CompilationUnit cu = StaticJavaParser.parse(file);

            final String[] lastFoundType = {null};
            final String[] lastFoundName = {null};
            final int[] lastFoundStartLine = {-1};
            final int[] lastFoundEndLine = {-1};

            TreeVisitor visitor = new TreeVisitor() {
                @Override
                public void process(Node node) {
                    if (node instanceof MethodDeclaration || node instanceof ClassOrInterfaceDeclaration) {
                        int startLine = node.getBegin().map(position -> position.line).orElse(-1);
                        int endLine = node.getEnd().map(position -> position.line).orElse(-1);

                        if (startLine <= lineNumber && lineNumber <= endLine) {
                            lastFoundType[0] = (node instanceof MethodDeclaration) ? "Method" : "Class";
                            lastFoundName[0] = (node instanceof MethodDeclaration) ? ((MethodDeclaration) node).getNameAsString() : ((ClassOrInterfaceDeclaration) node).getNameAsString();
                            lastFoundStartLine[0] = startLine;
                            lastFoundEndLine[0] = endLine;
                        }
                    }
                }
            };

            visitor.visitPreOrder(cu);

            if (lastFoundType[0] != null) {
                System.out.println(lastFoundStartLine[0] + " " + lastFoundEndLine[0]);
            } else {
                System.out.println(lineNumber + " " + lineNumber);
            }
        } catch (Exception e) {
            System.out.println("-1 -1");
        }
    }
}
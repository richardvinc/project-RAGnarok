import "./globals.css";

export const metadata = {
  title: "Project RAGnarok",
  description:
    "RAG inspector with structured LLM decision logging and tool execution tracing.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

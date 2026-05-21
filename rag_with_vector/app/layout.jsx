import "./globals.css";

export const metadata = {
  title: "Project The Jungle Book",
  description:
    "Implementation of LLM, RAG, and image generation by utilizing The Jungle Book from Gutenberg Project",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

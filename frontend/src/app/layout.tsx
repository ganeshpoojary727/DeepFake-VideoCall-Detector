import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "DeepGuard Forensics — AI Deepfake Detection & Forensic Intelligence",
  description:
    "Enterprise-grade deepfake detection platform powered by AASIST, EfficientNet-B4 spatiotemporal transformers, and multimodal forensic fusion. Analyze images, video, and audio for synthetic media artifacts with explainable AI diagnostics.",
  keywords: [
    "deepfake detection",
    "forensic AI",
    "media verification",
    "AASIST",
    "EfficientNet",
    "synthetic media",
    "audio spoofing",
    "video forensics",
  ],
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-[#07080f] text-[#e8eaed]">
        {children}
      </body>
    </html>
  );
}

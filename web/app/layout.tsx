import type { Metadata } from "next";
import { Geist_Mono, Oxanium, Noto_Serif } from "next/font/google";

import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { TopNav } from "@/components/navigation/TopNav";
import { MobileDock } from "@/components/navigation/MobileDock";
import { Toaster } from "@/components/ui/sonner";
import { cn } from "@/lib/utils";

const notoSerifHeading = Noto_Serif({
  subsets: ["latin"],
  variable: "--font-heading",
});

const oxanium = Oxanium({ subsets: ["latin"], variable: "--font-sans" });

const fontMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL?.trim() || "https://combina2.up.railway.app";
const siteTitle = "Combina2";
const siteDescription =
  "Genera o identifica paletas de colores tomando una imagen ✨";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: siteTitle,
    template: `%s | ${siteTitle}`,
  },
  description: siteDescription,
  applicationName: siteTitle,
  alternates: {
    canonical: "/",
  },
  openGraph: {
    type: "website",
    locale: "es_ES",
    url: "/",
    siteName: siteTitle,
    title: siteTitle,
    description: siteDescription,
    images: [
      {
        url: "/combina2.jpg",
        width: 1408,
        height: 768,
        alt: "Combina2: genera o identifica paletas de colores a partir de una imagen",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: siteTitle,
    description: siteDescription,
    images: ["/combina2.jpg"],
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="es"
      suppressHydrationWarning
      className={cn(
        "antialiased",
        fontMono.variable,
        "font-sans",
        oxanium.variable,
        notoSerifHeading.variable
      )}
    >
      <body>
        <ThemeProvider>
          <TopNav />
          <main className="mx-auto min-h-svh w-full max-w-6xl px-4 pb-24 pt-6 md:pb-6">
            {children}
          </main>
          <MobileDock />
          <Toaster richColors position="top-center" />
        </ThemeProvider>
      </body>
    </html>
  );
}

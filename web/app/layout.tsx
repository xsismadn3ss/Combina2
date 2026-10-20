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

export const metadata = {
  title: "Combina2",
  description: "Combina 2 o más colores para generar una paleta. Identifica paletas de colores tomando una foto.",
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

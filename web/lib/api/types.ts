export type ColorRole =
  | "primary"
  | "secondary"
  | "accent"
  | "complementary"
  | "analogous_1"
  | "analogous_2"
  | "triadic_1"
  | "triadic_2"
  | "predominant";

export type HarmonyType =
  | "complementario"
  | "triada"
  | "analogo"
  | "monocromatica";

export interface Color {
  value: string;
  role: string;
  weight?: number | null;
}

export interface ScoredPalette {
  type: string;
  colors: Color[];
  score: number;
}

export interface PaletteResponse {
  type: string;
  colors: Color[];
  fallback?: boolean;
}

export interface ExtractResponse {
  colors: Color[];
  fallback: boolean;
}

export interface GenerateResponse {
  palettes: ScoredPalette[];
  fallback: boolean;
}

export interface IdentifyResponse {
  type: string;
  confidence: number;
  colors: Color[];
  fallback: boolean;
}

export interface FeedbackRequest {
  palette: PaletteResponse;
  accepted: boolean;
  score?: number | null;
  imageSha256?: string | null;
}

export interface FeedbackResponse {
  ok: boolean;
}

export interface CreatePaletteRequest {
  colors: string[];
  harmony?: HarmonyType | null;
  topK?: number;
}

/**
 * Servicio centralizado de colores para la aplicación
 * Tema único: dark
 * Lee variables CSS desde :root en app/globals.css
 */

// ==========================================
// HELPERS
// ==========================================

const isBrowser = typeof window !== "undefined";

const readCssVar = (variableName: string, fallback: string): string => {
    if (!isBrowser) return fallback;

    const value = getComputedStyle(document.documentElement)
        .getPropertyValue(variableName)
        .trim();

    return value || fallback;
};

const resolveColor = (variableName: string, fallback: string): string => {
    return readCssVar(variableName, fallback);
};

// ==========================================
// PALETA DARK (FALLBACKS)
// ==========================================

export const DARK_COLORS = {
    // Primarios
    primary: "#60a5fa",
    primaryLight: "#93c5fd",
    primaryDark: "#1e40af",

    // Secundarios
    secondary: "#a78bfa",
    secondaryLight: "#c4b5fd",
    secondaryDark: "#5b21b6",

    // Neutros
    background: "#0f172a",
    surface: "#1e293b",
    surfaceAlt: "#334155",
    text: "#f1f5f9",
    textLight: "#cbd5e1",
    textLighter: "#94a3b8",
    border: "#334155",

    // Estados
    success: "#34d399",
    successLight: "#a7f3d0",
    successDark: "#059669",

    warning: "#fbbf24",
    warningLight: "#fcd34d",
    warningDark: "#b45309",

    error: "#f87171",
    errorLight: "#fecaca",
    errorDark: "#991b1b",

    info: "#06b6d4",
    infoLight: "#67e8f9",
    infoDark: "#0e7490",

    // Específicos de Audio
    alertable: "#f87171",
    noAlertable: "#60a5fa",
    audioSource: "#fbbf24",
    audioFormat: "#34d399",
    sampleRate: "#a78bfa",
} as const;

export type ColorKey = keyof typeof DARK_COLORS;

// ==========================================
// COLORES DESDE :ROOT
// ==========================================

export const getColor = (colorKey: ColorKey): string => {
    const cssVarMap: Record<ColorKey, string> = {
        primary: "--color-primary",
        primaryLight: "--color-primary-light",
        primaryDark: "--color-primary-dark",

        secondary: "--color-secondary",
        secondaryLight: "--color-secondary-light",
        secondaryDark: "--color-secondary-dark",

        background: "--color-background",
        surface: "--color-surface",
        surfaceAlt: "--color-surface-alt",
        text: "--color-text",
        textLight: "--color-text-light",
        textLighter: "--color-text-lighter",
        border: "--color-border",

        success: "--color-success",
        successLight: "--color-success-light",
        successDark: "--color-success-dark",

        warning: "--color-warning",
        warningLight: "--color-warning-light",
        warningDark: "--color-warning-dark",

        error: "--color-error",
        errorLight: "--color-error-light",
        errorDark: "--color-error-dark",

        info: "--color-info",
        infoLight: "--color-info-light",
        infoDark: "--color-info-dark",

        alertable: "--color-alertable",
        noAlertable: "--color-no-alertable",
        audioSource: "--color-audio-source",
        audioFormat: "--color-audio-format",
        sampleRate: "--color-sample-rate",
    };

    const cssVar = cssVarMap[colorKey];
    return resolveColor(cssVar, DARK_COLORS[colorKey]);
};

export const getThemeColors = () => {
    return {
        primary: getColor("primary"),
        primaryLight: getColor("primaryLight"),
        primaryDark: getColor("primaryDark"),

        secondary: getColor("secondary"),
        secondaryLight: getColor("secondaryLight"),
        secondaryDark: getColor("secondaryDark"),

        background: getColor("background"),
        surface: getColor("surface"),
        surfaceAlt: getColor("surfaceAlt"),
        text: getColor("text"),
        textLight: getColor("textLight"),
        textLighter: getColor("textLighter"),
        border: getColor("border"),

        success: getColor("success"),
        successLight: getColor("successLight"),
        successDark: getColor("successDark"),

        warning: getColor("warning"),
        warningLight: getColor("warningLight"),
        warningDark: getColor("warningDark"),

        error: getColor("error"),
        errorLight: getColor("errorLight"),
        errorDark: getColor("errorDark"),

        info: getColor("info"),
        infoLight: getColor("infoLight"),
        infoDark: getColor("infoDark"),

        alertable: getColor("alertable"),
        noAlertable: getColor("noAlertable"),
        audioSource: getColor("audioSource"),
        audioFormat: getColor("audioFormat"),
        sampleRate: getColor("sampleRate"),
    };
};

export const hexToRgba = (hex: string, alpha: number = 1): string => {
    const normalized = hex.replace("#", "").trim();

    if (normalized.length !== 6) {
        return hex;
    }

    const r = parseInt(normalized.slice(0, 2), 16);
    const g = parseInt(normalized.slice(2, 4), 16);
    const b = parseInt(normalized.slice(4, 6), 16);

    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

export const generateCSSVariables = (): string => {
    const colors = getThemeColors();

    return Object.entries(colors)
        .map(([key, value]) => `--color-${key.replace(/[A-Z]/g, match => `-${match.toLowerCase()}`)}: ${value};`)
        .join("\n  ");
};

export const COLOR_PALETTES = {
    status: {
        dark: {
            success: getColor("success"),
            warning: getColor("warning"),
            error: getColor("error"),
            info: getColor("info"),
        },
    },
    audio: {
        dark: {
            alertable: getColor("alertable"),
            noAlertable: getColor("noAlertable"),
            source: getColor("audioSource"),
            format: getColor("audioFormat"),
            sampleRate: getColor("sampleRate"),
        },
    },
    neutral: {
        dark: {
            background: getColor("background"),
            surface: getColor("surface"),
            text: getColor("text"),
            border: getColor("border"),
        },
    },
} as const;

export default {
    dark: DARK_COLORS,
    get: getColor,
    getTheme: getThemeColors,
    rgba: hexToRgba,
    css: generateCSSVariables,
    palettes: COLOR_PALETTES,
};
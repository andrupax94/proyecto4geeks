"use client";

import { useEffect, useState } from "react";
import { subscribeMovilVisible } from "@/services/movil_window_service";

export function useMobileVisibility() {
    const [isMobileVisible, setIsMobileVisible] = useState(false);

    useEffect(() => {
        const unsubscribe = subscribeMovilVisible(setIsMobileVisible);
        return unsubscribe;
    }, []);

    return isMobileVisible;
}
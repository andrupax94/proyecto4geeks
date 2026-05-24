// src/services/movi_window_service.ts

type Listener = (visible: boolean) => void;

let mobileVisible = false;
const listeners = new Set<Listener>();

export function showMovil() {
    mobileVisible = !mobileVisible;
    listeners.forEach((listener) => listener(mobileVisible));
    return mobileVisible;
}

export function getMovilVisible() {
    return mobileVisible;
}

export function subscribeMovilVisible(listener: Listener) {
    listeners.add(listener);
    listener(mobileVisible);

    return () => {
        listeners.delete(listener);
    };
}
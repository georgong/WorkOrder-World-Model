"use client";

import { createContext, useContext, useState, useCallback, type ReactNode } from "react";

interface HeaderState {
  hasResult: boolean;
  isDemo: boolean;
  onReset: () => void;
}

const HeaderContext = createContext<HeaderState>({
  hasResult: false,
  isDemo: false,
  onReset: () => {},
});

const SetHeaderContext = createContext<(s: Partial<HeaderState>) => void>(() => {});

export function HeaderProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<HeaderState>({
    hasResult: false,
    isDemo: false,
    onReset: () => {},
  });

  const update = useCallback(
    (partial: Partial<HeaderState>) => setState((prev) => ({ ...prev, ...partial })),
    []
  );

  return (
    <HeaderContext.Provider value={state}>
      <SetHeaderContext.Provider value={update}>
        {children}
      </SetHeaderContext.Provider>
    </HeaderContext.Provider>
  );
}

export function useHeaderActions() {
  return useContext(HeaderContext);
}

export function useSetHeader() {
  return useContext(SetHeaderContext);
}

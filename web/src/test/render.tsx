import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, type RenderOptions } from "@testing-library/react";
import type { ComponentProps, ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";

type MemoryRouterEntry = NonNullable<ComponentProps<typeof MemoryRouter>["initialEntries"]>[number];

export function renderWithProviders(
  ui: ReactElement,
  { route = "/", ...options }: { route?: MemoryRouterEntry } & Omit<RenderOptions, "wrapper"> = {},
) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
    options,
  );
}

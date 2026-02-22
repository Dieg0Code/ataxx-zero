import * as React from "react";
import { cn } from "@/shared/lib/utils";

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, type = "text", ...props }, ref) => {
    return (
      <input
        ref={ref}
        type={type}
        className={cn(
          "flex h-9 w-full rounded-md border border-line bg-bg1 px-3 py-1 text-sm text-textMain placeholder:text-textDim/70 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary",
          className,
        )}
        {...props}
      />
    );
  },
);
Input.displayName = "Input";

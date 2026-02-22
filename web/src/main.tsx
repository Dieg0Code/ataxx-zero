import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "@/app/App";
import "@/shared/styles/globals.css";
import "@/shared/styles/tokens.css";
import "@/shared/styles/themes/terminal-neo.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

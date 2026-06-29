import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "material-symbols/outlined.css";
import "./styles/tokens.css";
import "./styles/desktop.css";
import "./styles/mobile.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

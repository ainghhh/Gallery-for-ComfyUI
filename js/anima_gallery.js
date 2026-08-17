import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "Gallery4ComfyUI.Opener",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Gallery4ComfyUIOpener") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);

      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = "打开图库";
      btn.style.cssText = `
        width: 100%;
        padding: 8px 0;
        border: none;
        border-radius: 6px;
        background: #3b6ef5;
        color: #fff;
        font-size: 13px;
        font-family: inherit;
        cursor: pointer;
      `;

      const setLabel = (txt, disabled) => {
        btn.textContent = txt;
        btn.disabled = !!disabled;
        btn.style.opacity = disabled ? "0.6" : "1";
      };

      btn.addEventListener("click", async () => {
        const w = this.widgets?.find(x => x.name === "webui_url");
        const cfg = (w?.value || "").trim();
        const target = cfg || "/gallery4comfyui/";

        // 懒启动：先触发索引（幂等），再打开页面
        setLabel("正在启动图库…", true);
        try {
          await fetch("/gallery4comfyui/api/boot", { cache: "no-store" });
        } catch (e) { /* ComfyUI 未就绪时静默，页面打开后也会自动触发 */ }
        setLabel("打开图库", false);
        window.open(target, "_blank");
      });

      const dom = this.addDOMWidget("open_gallery_btn", "button", btn, { serialize: false });
      dom.computeSize = function () { return [this.parent?.size?.[0] || 200, 38]; };
      dom.computedHeight = 38;
      return r;
    };
  },
});

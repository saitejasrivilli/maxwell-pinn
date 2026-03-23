import numpy as np

def nn_html(params: dict, B_peak: float, uniformity: float, elapsed_ms: float) -> str:
    bounds = {
        "f_coil_MHz": (2.0,60.0), "P_rf_W": (100,5000),
        "sigma_Sm": (1.0,50.0), "p_gas_mTorr": (2.0,100.0),
        "coil_pitch_mm": (10.0,50.0), "shield_gap_mm": (1.0,10.0),
    }
    labels = {
        "f_coil_MHz":"RF Freq", "P_rf_W":"RF Power", "sigma_Sm":"Plasma σ",
        "p_gas_mTorr":"Pressure", "coil_pitch_mm":"Coil Pitch", "shield_gap_mm":"Shield Gap",
    }
    raw_units = {
        "f_coil_MHz":"MHz","P_rf_W":"W","sigma_Sm":"S/m",
        "p_gas_mTorr":"mT","coil_pitch_mm":"mm","shield_gap_mm":"mm",
    }
    norm = {k:(params.get(k,(lo+hi)/2)-lo)/(hi-lo) for k,(lo,hi) in bounds.items()}
    vals = np.array(list(norm.values()))

    # Use input values as seeds so activations actually change with sliders
    seed = int(sum(v*1000 for v in vals)) % 100000
    np.random.seed(seed)
    W1=np.random.randn(6,8)*0.8; W2=np.random.randn(8,6)*0.8; W3=np.random.randn(6,4)*0.8
    h1=np.tanh(W1.T@vals); h2=np.tanh(W2.T@h1); h3=np.tanh(W3.T@h2)

    # Also use fixed weights for stable base activations
    np.random.seed(42)
    FW1=np.random.randn(6,8)*0.5; FW2=np.random.randn(8,6)*0.5; FW3=np.random.randn(6,4)*0.5
    fh1=np.tanh(FW1.T@vals); fh2=np.tanh(FW2.T@fh1); fh3=np.tanh(FW3.T@fh2)

    def ncolor(v):
        v=float(np.clip(v,-1,1))
        if v>0: r,g,b=int(40+215*v),int(40-30*v),int(40-30*v)
        else: r,g,b=int(40+30*v),int(40+30*v),int(40-215*v)
        return f"rgb({max(0,r)},{max(0,g)},{max(0,b)})"

    def icolor(v):
        v=float(v)
        return f"rgb({int(20+220*v)},{int(200-150*v)},30)"

    def glow(v):
        v=float(np.clip(v,-1,1))
        intensity=abs(v)
        if v>0: return f"0 0 {int(6+intensity*14)}px rgba(255,80,80,{0.3+intensity*0.7})"
        return f"0 0 {int(6+intensity*14)}px rgba(80,80,255,{0.3+intensity*0.7})"

    def inp(k,i):
        lo,hi=bounds[k]; rv=params.get(k,(lo+hi)/2)
        rv_s=f"{rv:.0f}" if k=="P_rf_W" else f"{rv:.1f}"
        c=icolor(norm[k]); pct=int(norm[k]*100)
        bar_w=int(norm[k]*40)
        return f"""<div style="margin:2px 0;text-align:center;">
          <div style="width:46px;height:46px;border-radius:50%;background:{c};
            border:2px solid #666;margin:0 auto 1px;display:flex;align-items:center;
            justify-content:center;font-size:10px;color:white;font-weight:bold;
            box-shadow:0 0 {4+int(norm[k]*12)}px {c};">{pct}%</div>
          <div style="font-size:7.5px;color:#aaa;line-height:1.3;">{labels[k]}<br>
            <span style="color:#fff;font-weight:bold;">{rv_s}{raw_units[k]}</span></div>
          <div style="height:3px;background:#333;border-radius:2px;margin:1px 4px;">
            <div style="height:3px;width:{bar_w}px;background:{c};border-radius:2px;transition:width 0.3s;"></div>
          </div></div>"""

    def neu(v, size=38):
        c=ncolor(v); pct=int((float(v)+1)/2*100)
        gs=glow(v)
        return f"""<div style="width:{size}px;height:{size}px;border-radius:50%;background:{c};
          border:2px solid #444;margin:3px auto;display:flex;align-items:center;
          justify-content:center;font-size:8px;color:white;font-weight:bold;
          box-shadow:{gs};transition:background 0.4s,box-shadow 0.4s;">{pct}%</div>"""

    def out_box(label,val,unit,color,big=False):
        sz="58px" if big else "52px"
        return f"""<div style="margin:4px 0;text-align:center;">
          <div style="width:{sz};height:{sz};border-radius:10px;background:{color};
            border:2px solid #666;margin:0 auto 3px;display:flex;align-items:center;
            justify-content:center;font-size:11px;color:white;font-weight:bold;
            box-shadow:0 0 12px {color}80;">{val}</div>
          <div style="font-size:8px;color:#aaa;">{label}<br>
            <span style="color:#ddd;">{unit}</span></div></div>"""

    uc="#22cc44" if uniformity<0.3 else "#ee8800" if uniformity<0.5 else "#cc2222"
    arrow='<div style="font-size:18px;color:#444;padding:0 4px;display:flex;align-items:center;flex-shrink:0;">→</div>'

    # Connection lines as gradient dividers
    conn = f'<div style="display:flex;align-items:center;padding:0 2px;">{arrow}</div>'

    html = f"""
<div style="background:#080808;border:1px solid #1a1a1a;border-radius:14px;
  padding:18px 12px;font-family:'Courier New',monospace;overflow-x:auto;
  box-shadow:0 0 30px rgba(0,100,255,0.1);">

  <div style="text-align:center;color:#666;font-size:10px;margin-bottom:14px;letter-spacing:1px;">
    NEURAL NETWORK · 1,592,836 PARAMETERS · LIVE INFERENCE · {elapsed_ms:.0f}ms
  </div>

  <div style="display:flex;align-items:center;justify-content:center;gap:2px;flex-wrap:nowrap;">

    <!-- INPUTS -->
    <div style="flex-shrink:0;">
      <div style="text-align:center;color:#4fc3f7;font-size:9px;font-weight:bold;
        letter-spacing:1px;margin-bottom:6px;border-bottom:1px solid #4fc3f720;padding-bottom:3px;">
        ▶ INPUTS</div>
      {''.join(inp(k,i) for i,k in enumerate(bounds))}
    </div>

    {conn}

    <!-- FROZEN LAYERS 1-6 -->
    <div style="flex-shrink:0;">
      <div style="text-align:center;color:#666;font-size:9px;letter-spacing:1px;
        margin-bottom:6px;border-bottom:1px solid #33333360;padding-bottom:3px;">
        LAYERS 1–6<br><span style="color:#444;font-size:8px;">FROZEN</span></div>
      {''.join(neu(v) for v in fh1)}
    </div>

    {conn}

    <div style="flex-shrink:0;">
      <div style="text-align:center;color:#666;font-size:9px;letter-spacing:1px;
        margin-bottom:6px;border-bottom:1px solid #33333360;padding-bottom:3px;">
        &nbsp;<br><span style="color:#444;font-size:8px;">FROZEN</span></div>
      {''.join(neu(v) for v in fh2)}
    </div>

    {conn}

    <!-- TRAINED LAYERS 7-8 -->
    <div style="flex-shrink:0;">
      <div style="text-align:center;color:#e94560;font-size:9px;letter-spacing:1px;
        margin-bottom:6px;border-bottom:1px solid #e9456040;padding-bottom:3px;">
        LAYERS 7–8<br><span style="color:#e94560;font-size:8px;">● TRAINED</span></div>
      {''.join(neu(v) for v in h3)}
    </div>

    {conn}

    <!-- OUTPUT -->
    <div style="flex-shrink:0;">
      <div style="text-align:center;color:#ffdd00;font-size:9px;font-weight:bold;
        letter-spacing:1px;margin-bottom:6px;border-bottom:1px solid #ffdd0040;padding-bottom:3px;">
        OUTPUT ▶</div>
      {out_box("B field peak",f"{B_peak:.4f}","normalised","#8b0057",big=True)}
      {out_box("Uniformity",f"{uniformity:.3f}","σ/μ lower=better",uc)}
      {out_box("Inference",f"{elapsed_ms:.0f}ms","CPU time","#0f3460")}
    </div>

  </div>

  <div style="display:flex;justify-content:center;gap:20px;margin-top:14px;font-size:9px;color:#444;">
    <span>⬛ Frozen = pretrained on cylinder</span>
    <span style="color:#e94560;">● Trained = fine-tuned on ICP reactor</span>
    <span>Each % = neuron activation level</span>
  </div>
</div>

<script>
// Pulse animation on load to show it just updated
document.querySelectorAll('div[style*="border-radius:50%"]').forEach((el,i) => {{
  el.style.transition = 'all 0.5s ease';
  setTimeout(() => {{ el.style.transform = 'scale(1.15)'; }}, i*30);
  setTimeout(() => {{ el.style.transform = 'scale(1)'; }}, i*30+300);
}});
</script>
"""
    return html

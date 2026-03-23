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
        "f_coil_MHz":"MHz", "P_rf_W":"W", "sigma_Sm":"S/m",
        "p_gas_mTorr":"mT", "coil_pitch_mm":"mm", "shield_gap_mm":"mm",
    }
    norm = {k: (params.get(k,(lo+hi)/2)-lo)/(hi-lo) for k,(lo,hi) in bounds.items()}
    vals = list(norm.values())
    np.random.seed(42)
    W1=np.random.randn(6,8)*0.5; W2=np.random.randn(8,6)*0.5; W3=np.random.randn(6,4)*0.5
    x=np.array(vals); h1=np.tanh(W1.T@x); h2=np.tanh(W2.T@h1); h3=np.tanh(W3.T@h2)

    def ncolor(v):
        v=float(np.clip(v,-1,1))
        if v>0: return f"rgb({int(80+175*v)},{int(80-60*v)},{int(80-60*v)})"
        return f"rgb({int(80+60*v)},{int(80+60*v)},{int(80-175*v)})"

    def icolor(v):
        v=float(v); return f"rgb({int(30+200*v)},{int(180-100*v)},50)"

    def inp(k):
        lo,hi=bounds[k]; rv=params.get(k,(lo+hi)/2)
        rv_s=f"{rv:.1f}" if isinstance(rv,float) else str(rv)
        c=icolor(norm[k]); pct=int(norm[k]*100)
        return f"""<div style="margin:3px 0;text-align:center;">
          <div style="width:44px;height:44px;border-radius:50%;background:{c};border:2px solid #555;
            margin:0 auto 2px;display:flex;align-items:center;justify-content:center;
            font-size:9px;color:white;font-weight:bold;">{pct}%</div>
          <div style="font-size:8px;color:#aaa;line-height:1.2;">{labels[k]}<br>
            <span style="color:white;font-weight:bold;">{rv_s}{raw_units[k]}</span></div></div>"""

    def neu(v):
        c=ncolor(v); pct=int((float(v)+1)/2*100)
        return f"""<div style="width:36px;height:36px;border-radius:50%;background:{c};
          border:2px solid #333;margin:3px auto;display:flex;align-items:center;
          justify-content:center;font-size:8px;color:white;font-weight:bold;">{pct}%</div>"""

    def out(label,val,unit,color):
        return f"""<div style="margin:4px 0;text-align:center;">
          <div style="width:52px;height:52px;border-radius:8px;background:{color};
            border:2px solid #555;margin:0 auto 3px;display:flex;align-items:center;
            justify-content:center;font-size:10px;color:white;font-weight:bold;">{val}</div>
          <div style="font-size:8px;color:#aaa;">{label}<br>{unit}</div></div>"""

    uc = "#22aa44" if uniformity<0.3 else "#ee8800" if uniformity<0.5 else "#cc2222"
    arrow = '<div style="font-size:20px;color:#555;padding:0 6px;display:flex;align-items:center;">→</div>'

    return f"""<div style="background:#0e0e0e;border:1px solid #222;border-radius:12px;padding:16px;font-family:monospace;">
  <div style="text-align:center;color:#888;font-size:11px;margin-bottom:10px;">
    NEURAL NETWORK — 1,592,836 parameters — move a slider to see neurons update
  </div>
  <div style="display:flex;align-items:center;justify-content:center;">
    <div><div style="text-align:center;color:#4fc3f7;font-size:10px;margin-bottom:4px;font-weight:bold;">INPUTS<br>(6 params)</div>
      {''.join(inp(k) for k in bounds)}</div>
    {arrow}
    <div><div style="text-align:center;color:#888;font-size:10px;margin-bottom:4px;">Layers 1-6<br><span style="color:#aaa">(frozen)</span></div>
      {''.join(neu(v) for v in h1)}</div>
    {arrow}
    <div><div style="text-align:center;color:#888;font-size:10px;margin-bottom:4px;">Layers 2-6<br><span style="color:#aaa">(frozen)</span></div>
      {''.join(neu(v) for v in h2)}</div>
    {arrow}
    <div><div style="text-align:center;color:#e94560;font-size:10px;margin-bottom:4px;">Layers 7-8<br><span style="color:#e94560">(trained)</span></div>
      {''.join(neu(v) for v in h3)}</div>
    {arrow}
    <div><div style="text-align:center;color:#ffdd00;font-size:10px;margin-bottom:4px;font-weight:bold;">OUTPUT</div>
      {out("B peak",f"{B_peak:.3f}","norm.","#c03080")}
      {out("Uniformity",f"{uniformity:.2f}","σ/μ",uc)}
      {out("Time",f"{elapsed_ms:.0f}","ms","#0f3460")}</div>
  </div>
  <div style="text-align:center;color:#444;font-size:10px;margin-top:8px;">
    Grey = frozen (pretrained on cylinder) · Red = fine-tuned on ICP reactor
  </div>
</div>"""

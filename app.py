import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from plotly.colors import sample_colorscale

st.set_page_config(layout="wide", page_title="Estroncas & Longarinas")
st.title("Escoramento em planta – Estroncas e Longarinas")

# ---------------- PARÂMETROS ----------------
with st.sidebar:
    st.header("Parâmetros do nível")
    q   = st.number_input("Reação do nível q (kN/m)", value=120.0)
    E   = st.number_input("E (GPa)", value=200.0) * 1e6
    Aw  = st.number_input("Longarina – A (cm²)", value=150.0) / 1e4
    Iw  = st.number_input("Longarina – I plano horizontal (cm⁴)", value=60000.0) / 1e8
    As  = st.number_input("Estronca – A (cm²)", value=120.0) / 1e4
    ft  = st.number_input("Fator de temperatura (N)", value=1.10)
    esc = st.slider("Escala do diagrama de M", 0.0, 0.1, 0.02, 0.005)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Contorno da cava (vértices)")
    st.caption("q_lado vazio = usa q global. Lado i liga vértice i → i+1.")
    dfP = st.data_editor(pd.DataFrame({
        "x": [0., 40, 40, 30, 30, 0], "y": [0., 0, 18, 18, 22, 22],
        "q_lado": [np.nan]*6}), num_rows="dynamic", key="poly")
with c2:
    st.subheader("Estroncas")
    st.caption("Extremidades devem estar sobre o contorno. Desmarque 'ativa' p/ perda.")
    dfS = st.data_editor(pd.DataFrame({
        "x1": [10., 20, 30, 0, 36, 0], "y1": [0., 0, 0, 4, 0, 18],
        "x2": [10., 20, 30, 4, 40, 4], "y2": [22., 22, 18, 0, 4, 22],
        "ativa": [True]*6}), num_rows="dynamic", key="str")

# ---------------- SOLVER ----------------
def solve(P, qs, S):
    n = len(P)
    area = 0.5*sum(P[i][0]*P[(i+1)%n][1] - P[(i+1)%n][0]*P[i][1] for i in range(n))
    sg = 1 if area > 0 else -1               # garante carga p/ dentro da cava
    nodes = {}
    key = lambda p: (round(float(p[0]),3), round(float(p[1]),3))
    nid = lambda p: nodes.setdefault(key(p), len(nodes))
    ends = [s[:2] for s in S] + [s[2:] for s in S]
    elems, onwall = [], set()
    for e in range(n):
        a, b = np.array(P[e], float), np.array(P[(e+1)%n], float)
        d = b - a; ts = {0.0, 1.0}
        for p in ends:
            p = np.array(p, float); t = (p-a)@d/(d@d)
            if -1e-9 <= t <= 1+1e-9 and np.linalg.norm(a+t*d-p) < 1e-3:
                ts.add(min(max(t, 0), 1)); onwall.add(key(p))
        ts = sorted(ts)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            if t1 - t0 > 1e-9:
                elems.append((nid(a+t0*d), nid(a+t1*d), 'L', sg*qs[e], e))
    bad = [k for k, s in enumerate(S) if key(s[:2]) not in onwall or key(s[2:]) not in onwall]
    for k, s in enumerate(S):
        elems.append((nid(s[:2]), nid(s[2:]), 'E', 0.0, k))

    XY = np.array(sorted(nodes, key=nodes.get), float)
    nd = 3*len(XY); K = np.zeros((nd, nd)); F = np.zeros(nd); data = []
    for i, j, tp, ql, tag in elems:
        d = XY[j]-XY[i]; L = np.linalg.norm(d); c, s = d/L
        r = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]); T = np.kron(np.eye(2), r)
        A, I = (Aw, Iw) if tp == 'L' else (As, 0.0)
        k = np.zeros((6, 6)); ea = E*A/L
        k[0,0] = k[3,3] = ea; k[0,3] = k[3,0] = -ea
        if I:
            a_, b_, c_ = 12*E*I/L**3, 6*E*I/L**2, E*I/L
            ix = [1, 2, 4, 5]
            k[np.ix_(ix, ix)] = [[a_, b_, -a_, b_], [b_, 4*c_, -b_, 2*c_],
                                 [-a_, -b_, a_, -b_], [b_, 2*c_, -b_, 4*c_]]
        fef = np.array([0, ql*L/2, ql*L**2/12, 0, ql*L/2, -ql*L**2/12])
        dof = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
        K[np.ix_(dof, dof)] += T.T@k@T; F[dof] += T.T@fef
        data.append(dict(i=i, j=j, tp=tp, q=ql, tag=tag, L=L, T=T, k=k, fef=fef, dof=dof))
    K += np.eye(nd)*1.0                      # molas fracas (estabilidade)
    U = np.linalg.solve(K, F)
    for el in data:
        el["f"] = el["k"]@(el["T"]@U[el["dof"]]) - el["fef"]
    return XY, data, bad

P  = dfP[["x", "y"]].dropna().values.tolist()
qs = [q if pd.isna(v) else v for v in dfP["q_lado"].iloc[:len(P)]]
dfA = dfS[dfS["ativa"]].dropna()
S  = dfA[["x1", "y1", "x2", "y2"]].values.tolist()
XY, data, bad = solve(P, qs, S)
if bad:
    st.warning(f"Estroncas fora do contorno: {[int(dfA.index[b]) for b in bad]}")

# ---------------- RESULTADOS ----------------
rs, rw = [], []
for el in data:
    f, L = el["f"], el["L"]
    if el["tp"] == 'E':
        rs.append(dict(ID=int(dfA.index[el["tag"]]), L_m=round(L, 2),
                       N_kN=round(f[0], 1), N_temp_kN=round(f[0]*ft, 1)))
    else:
        x = np.linspace(0, L, 41)
        M = -f[2] + f[1]*x + el["q"]*x**2/2
        el["x"], el["M"] = x, M
        rw.append(dict(lado=el["tag"], de=tuple(XY[el["i"]]), para=tuple(XY[el["j"]]),
                       L_m=round(L, 2), N_kN=round(f[0], 1),
                       Vmax_kN=round(max(abs(f[1]), abs(f[4])), 1),
                       Mmax_kNm=round(np.abs(M).max(), 1)))
rs, rw = pd.DataFrame(rs), pd.DataFrame(rw)

# ---------------- DESENHO ----------------
fig = go.Figure()
Nmax = max(rs["N_kN"].abs().max(), 1) if len(rs) else 1
for el in data:
    a, b = XY[el["i"]], XY[el["j"]]
    if el["tp"] == 'L':
        fig.add_trace(go.Scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines",
                      line=dict(color="black", width=5), showlegend=False, hoverinfo="skip"))
        d = (b-a)/el["L"]; nrm = np.array([-d[1], d[0]])
        pts = a + np.outer(el["x"], d) + np.outer(el["M"]*esc, nrm)
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines",
                      line=dict(color="red", width=1.5), showlegend=False,
                      text=[f"M={m:.1f} kN·m" for m in el["M"]], hoverinfo="text"))
    else:
        N = el["f"][0]
        cor = sample_colorscale("Turbo", [min(abs(N)/Nmax, 1)])[0]
        fig.add_trace(go.Scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines",
                      line=dict(color=cor, width=6), showlegend=False,
                      hovertext=f"N = {N:.1f} kN", hoverinfo="text"))
        m = (a+b)/2
        fig.add_annotation(x=m[0], y=m[1], text=f"E{int(dfA.index[el['tag']])}<br>{N:.0f} kN",
                           showarrow=False, bgcolor="white", font=dict(size=11))
fig.update_layout(height=600, yaxis=dict(scaleanchor="x"), plot_bgcolor="#fdf6dc",
                  title="Planta – estroncas (cor = N) e momento nas longarinas (vermelho)")
st.plotly_chart(fig, use_container_width=True)

t1, t2 = st.columns(2)
with t1:
    st.subheader("Estroncas (+ compressão)")
    st.dataframe(rs, use_container_width=True)
    st.download_button("CSV estroncas", rs.to_csv(index=False), "estroncas.csv")
with t2:
    st.subheader("Longarinas (por trecho)")
    st.dataframe(rw, use_container_width=True)
    st.download_button("CSV longarinas", rw.to_csv(index=False), "longarinas.csv")

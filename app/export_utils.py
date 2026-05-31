import io
import datetime

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def _get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def export_to_csv(results):
    rows = []
    for i, r in enumerate(results):
        row = {
            'Khuôn mặt': f'Khuôn mặt {i+1}',
            'Biểu cảm': r['emotion'],
            'Độ tin cậy (%)': round(r.get('confidence', 0), 2),
        }
        for em, prob in r.get('probabilities', {}).items():
            row[f'{em} (%)'] = round(prob, 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding='utf-8-sig')
    buf.seek(0)
    return buf, f"ket_qua_{_get_timestamp()}.csv"


def export_to_image(annotated_image, results):
    h, w = annotated_image.shape[:2]
    summary_h = 150
    summary = np.ones((summary_h, w, 3), dtype=np.uint8) * 30

    counts = {}
    for r in results:
        em = r['emotion']
        counts[em] = counts.get(em, 0) + 1

    cv2.putText(summary, f"Tong so khuon mat: {len(results)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)
    y = 60
    for em, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        cv2.putText(summary, f"  {em}: {cnt}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25

    composite = np.vstack([annotated_image, summary])
    _, buffer = cv2.imencode('.png', cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    buf = io.BytesIO(buffer.tobytes())
    return buf, f"ket_qua_{_get_timestamp()}.png"


def export_to_pdf(results, total_faces):
    counts = {}
    for r in results:
        em = r['emotion']
        counts[em] = counts.get(em, 0) + 1

    emotions_sorted = sorted(counts.keys())
    values_sorted = [counts[e] for e in emotions_sorted]

    fig = Figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('#f5f5f5')

    ax_title = fig.add_axes([0.1, 0.92, 0.8, 0.05])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'Báo cáo phân tích biểu cảm khuôn mặt',
                  ha='center', va='center', fontsize=18, fontweight='bold')

    timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    ax_info = fig.add_axes([0.1, 0.86, 0.8, 0.05])
    ax_info.axis('off')
    ax_info.text(0.5, 0.5, f'Tổng số khuôn mặt: {total_faces} | Ngày: {timestamp}',
                 ha='center', va='center', fontsize=11, color='gray')

    ax_bar = fig.add_axes([0.1, 0.55, 0.8, 0.28])
    colors = plt.cm.Set2(np.linspace(0, 1, len(emotions_sorted)))
    bars = ax_bar.bar(emotions_sorted, values_sorted, color=colors, edgecolor='white')
    ax_bar.set_ylabel('Số lượng', fontsize=11)
    ax_bar.tick_params(labelsize=9)
    for bar, v in zip(bars, values_sorted):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(v), ha='center', va='bottom', fontsize=10)

    ax_pie = fig.add_axes([0.1, 0.25, 0.8, 0.28])
    wedges, texts, autotexts = ax_pie.pie(
        values_sorted, labels=emotions_sorted, autopct='%1.1f%%',
        colors=colors, startangle=140
    )
    for t in texts + autotexts:
        t.set_fontsize(9)

    ax_footer = fig.add_axes([0.1, 0.02, 0.8, 0.05])
    ax_footer.axis('off')
    ax_footer.text(0.5, 0.5, 'FaceE - Nhận diện biểu cảm khuôn mặt',
                   ha='center', va='center', fontsize=9, color='gray')

    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf, f"bao_cao_{_get_timestamp()}.pdf"

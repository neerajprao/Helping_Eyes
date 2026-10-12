"""
Draws docs/architecture.png (and .svg): the life cycle of a request, from input to output.
Edit the boxes and arrows below when the architecture changes, then run:

    python docs/make_architecture.py

Needs only the Python standard library, plus Chrome or Chromium for the PNG.
"""
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
W,H=1500,930
o=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
'<defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#374151"/></marker></defs>',
f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
'<text x="750" y="42" text-anchor="middle" font-size="28" font-weight="700" fill="#111827">Helping Eyes: life cycle of a request</text>']
def panel(x,y,w,h,t,c):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{c}" stroke="#9ca3af"/>')
    o.append(f'<text x="{x+w/2}" y="{y+28}" text-anchor="middle" font-size="18" font-weight="700" fill="#111827">{t}</text>')
def box(x,y,w,h,lines,fill="#ffffff",stroke="#4b5563"):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    n=len(lines); top=y+h/2-(n-1)*10+5
    for i,l in enumerate(lines):
        o.append(f'<text x="{x+w/2}" y="{top+i*20}" text-anchor="middle" font-size="{15 if i==0 else 13}" font-weight="{700 if i==0 else 400}" fill="#111827">{l}</text>')
def arrow(pts,label=None,lp=None,dash=False):
    d="M"+" L".join(f"{a},{b}" for a,b in pts)
    o.append(f'<path d="{d}" fill="none" stroke="#374151" stroke-width="1.8" marker-end="url(#a)"'+(' stroke-dasharray="6 4"' if dash else '')+'/>')
    if label:
        x,y=lp
        o.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="12" fill="#374151" stroke="#fff" stroke-width="4" paint-order="stroke">{label}</text>')
panel(30,70,700,675,"Camera path","#eff6ff")
panel(770,70,700,675,"Voice / typing path","#fff7ed")
I="#1d4ed8"; V="#c2410c"
# camera path
box(210,105,320,50,["INPUT 1: camera (or its shutter button)","or an uploaded photo"],"#dbeafe",I)
box(210,190,320,80,["vision.py, part D (the coach)","looks for text, checks blur and glare","a sheet that fills the picture is close enough","(uploads: shown on the photo too)"],"#fff",I)
box(45,190,140,84,["Hint","left, right, hold still,","move closer only","if text is unclear"],"#fff",I)
box(160,330,240,50,["Full photo","normal mode: text is steady"],"#fff",I)
box(160,420,240,60,["vision.py, part B","fix tilt, glare, shadows"],"#fff",I)
box(160,520,240,60,["vision.py, part A","picture to text (Apple Vision)"],"#fff",I)
box(440,330,260,50,["Full photo of the page","book mode: a page was turned"],"#fff",I)
box(440,420,260,118,["vision.py, part C (book mode)","split pages if a spine is found,","page number, paragraph order,","AI groups the lines, sidebars last,","a nudge is not a page turn"],"#fff",I)
box(440,550,260,55,["Page text","+ where each word is"],"#fff",I)
box(210,665,320,60,["Captured text","saved for questions"],"#e0e7ff",I)
# voice path
box(800,105,290,50,["INPUT 2: your voice or typing"],"#ffedd5",V)
box(800,195,290,60,["Browser: speech to text"],"#fff",V)
box(800,285,290,70,["commands.py","command or question?","unclear wording: the AI decides"],"#fff",V)
box(1180,295,260,60,["Do it","stop, repeat, next, book mode"],"#fff",V)
box(800,420,290,100,["assistant.py + the chosen AI model","Gemini (online) or Qwen 3B (Ollama)","answers only from the saved text","(book mode: also groups the lines)"],"#fff",V)
box(1180,420,260,90,["DuckDuckGo search","then the AI answers from","the results"],"#fff",V)
box(800,575,640,55,["Answer, sentence by sentence","each sentence goes to the voice (commands.py: local Kokoro voice)"],"#fff",V)
# output
box(30,800,1440,95,["OUTPUT: spoken aloud and shown on screen","spoken by the local Kokoro voice (on this Mac); backups: Edge voice, then the browser voice","(book mode also highlights the word being read)"],"#dcfce7","#15803d")
# arrows: camera path
arrow([(370,155),(370,195)])
arrow([(210,230),(185,230)])
arrow([(370,265),(280,330)],"normal",(300,298))
arrow([(370,265),(570,330)],"book mode",(500,298))
arrow([(280,380),(280,420)]); arrow([(280,480),(280,520)])
arrow([(570,380),(570,420)])
arrow([(440,450),(400,450)],"uses",(420,440),True)
arrow([(440,495),(400,540)],None,None,True)
arrow([(570,538),(570,550)])
arrow([(280,580),(300,665)]); arrow([(570,605),(470,665)])
arrow([(115,274),(115,800)])
arrow([(700,578),(750,578),(750,800)],"read the page",(716,640))
# voice path
arrow([(945,155),(945,195)]); arrow([(945,255),(945,285)])
arrow([(945,355),(945,420)],"question",(985,392))
arrow([(1090,325),(1180,325)],"command",(1135,316))
arrow([(1440,325),(1458,325),(1458,800)])
arrow([(1090,465),(1180,465)],"not in text + yes",(1135,452))
arrow([(945,510),(945,575)],"answer is in the text",(1030,548))
arrow([(1310,510),(1310,575)])
arrow([(1120,630),(1120,800)])
# saved text feeds the AI
arrow([(530,695),(780,695),(780,470),(800,470)],"saved text",(655,685))
o.append('</svg>')
svg_path = os.path.join(HERE, "architecture.svg")
open(svg_path, "w").write("\n".join(o))
print("wrote", svg_path)

# PNG: render the SVG with a headless Chrome / Chromium (any that is installed)
png_path = os.path.join(HERE, "architecture.png")
browsers = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium", "chromium-browser"]
chrome = next((b for b in browsers if shutil.which(b)), None)
if chrome is None:
    print("No Chrome / Chromium found: open architecture.svg in a browser and export it as architecture.png")
else:
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
                    "--force-device-scale-factor=1.5", f"--screenshot={png_path}", "file://" + svg_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print("wrote", png_path)

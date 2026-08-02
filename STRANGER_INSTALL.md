# Run NEURON on your computer

You get a private AI chat that runs on your own machine. In return, your computer helps a
shared network when you're not using it, and you earn **NRN** for that.

**Fair warning first:** NRN has no cash value — it's a record of what your machine contributed.
The network is small right now, so the numbers are small. Nothing you type ever leaves your
computer, and nothing is collected about you.

---

### Step 1 — Install Python

Go to **python.org/downloads**, get **Python 3.11 or newer**, and run it.

On Windows, tick **"Add Python to PATH"** on the first screen. It's easy to miss.

### Step 2 — Download NEURON

Go to **https://github.com/raman011sharma-code/neuron-network**, click the green **Code** button, then
**Download ZIP**. Unzip it somewhere you'll find again, like your Desktop.

### Step 3 — Start it

Open a terminal (Windows: search for **PowerShell**) and type these one at a time:

```
cd Desktop\neuron-network-main
python -m venv .venv
.venv\Scripts\activate
pip install -r agent/requirements.txt
python agent/agent.py
```

The fourth line takes a few minutes. The last line downloads about 1.4 GB — that's your piece
of the AI model. Leave the window open.

### Step 4 — Sign in

Open **http://localhost:8080** in your browser and sign in with Google or GitHub.

### Step 5 — Use it, and watch it earn

Ask it anything. Your balance is at the top of the page. Your computer is checked automatically
about a minute after it joins, and starts earning after that.

---

**To stop:** close the terminal window, or press `Ctrl` and `C` in it.

**To remove it completely:** `python agent/uninstall.py`

**If something goes wrong,** send me the file `agent/agent.log` from the folder you unzipped —
it says what happened.

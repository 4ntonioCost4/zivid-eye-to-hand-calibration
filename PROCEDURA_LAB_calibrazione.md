# Calibrazione camera Zivid ↔ FR3 con la board: procedura in laboratorio

Script: `auto_calibration_board.py` (repo `zivid-eye-to-hand-calibration`)
Risultato: `base_T_cam` (posa della camera nella base del robot, **in mm**)

---

## 0. Preparazione (ogni volta)

**Fisico**
- [ ] Board Zivid montata **rigidamente** sul gripper. Se è tenuta dalle dita: gripper **chiuso** e nessuno lo comanda (non lanciare `gripper_node.py`).
- [ ] In Desk, configurazione dell'end-effector **coerente** con quello montato (massa corretta per la gravity compensation).
- [ ] Area di lavoro libera, **pulsante di emergenza in mano**.

**PC del controllo** (pacmate-MD34400, container `franka_ros2_humble`)
- [ ] Driver avviato.
- [ ] Controller attivi: `ros2 control list_controllers -c /robot1/controller_manager`, con `torque_subscriber_controller` **active** (il robot si muove a mano).
- [ ] Se gli spawner vanno in timeout: Ctrl+C, verifica che non ci siano processi rimasti (`ps aux | grep -E "ros2_control_node|spawner"`) e rilancia.

**PC GPU** (pacmate22): un terminale
```bash
cd ~/zivid_ws/zivid-eye-to-hand-calibration && git pull
source /opt/ros/humble/setup.bash && source ~/venvs/zivid/bin/activate
ros2 topic list | grep current_pose      # deve comparire /robot1/franka_robot_state_broadcaster/current_pose
```
- [ ] Zivid Studio e nodo camera di H2 **chiusi** (la camera può essere usata da un solo programma).

---

## 1. Prima volta: `record` (guidi tu il robot)

```bash
python3 auto_calibration_board.py record --poses calibration_poses.yaml
```

| Vedi | Fai |
|---|---|
| `[camera] connected` e `RECORD mode` | aspetti |
| prompt `[0 cal / 0 ver] >` | porti il robot a mano nella posa, **togli le mani** |
| — | **INVIO** = posa di calibrazione |
| `saved cal01` | posa successiva |
| `NOT saved: robot moved during the capture` | aspetta che sia fermo, ripremi INVIO |
| `NOT saved: calibration board not detected` | cambia leggermente la posa e riprova |
| — | ripeti fino a **15–20 pose di calibrazione** |
| — | poi **4–5 pose di verifica**: **`v` + INVIO** |
| — | **`q` + INVIO** per finire |
| riquadro `SCORE ... RESULT: PASS / FAIL` | leggi il risultato |
| `Write these poses into calibration_poses.yaml? [y/N]` (solo se PASS) | **`y` + INVIO** |

**Come scegliere le pose**
- Board sempre rivolta verso la camera, a 35–90 cm da essa.
- Rotazioni ampie del polso (±30–40°) su assi diversi, non solo traslazioni.
- Posizioni e altezze diverse sopra l'area di lavoro, dove poi si faranno le prese.
- Registrale **in ordine**, con passaggi brevi tra una e l'altra: è il percorso che il robot rifarà da solo.

Se il risultato è **PASS**, fai commit del file delle pose:
```bash
git add calibration_poses.yaml && git commit -m "Pose di calibrazione salvate" && git push
```

---

## 2. Volte successive: `run` (automatico)

```bash
python3 auto_calibration_board.py check --poses calibration_poses.yaml
python3 auto_calibration_board.py run   --poses calibration_poses.yaml --gain-scale 0.5 --vmax 0.1
```
(`--gain-scale 0.5 --vmax 0.1` solo le prime volte; poi si possono togliere.)

| Vedi | Fai |
|---|---|
| `check`: tutte le pose `OK` e la durata stimata | se una è `BAD`, correggi o rifai `record` |
| `Joint states received at ... Hz` | sotto 300 Hz lo script si ferma da solo |
| `Distance to the first pose: ... rad` | controlla che sia ragionevole |
| `ENTER to start` | emergenza in mano, area libera → **INVIO** |
| `[i/N] calXX: moving ...` → `captured calXX` | **guardi e basta** |
| `skipped: ...` | una posa scartata, lo script continua |
| riquadro `SCORE ... PASS / FAIL` | leggi il risultato |

**Ctrl+C in qualunque momento** → il robot torna in gravity compensation (resta dov'è).

---

## 3. Risultato

Cartella `~/zivid_ws/calib_board_<data_ora>/`:
- `T_cam_to_base_Zivid_board.npy` → **matrice da usare** (mm)
- `report.json` → punteggio completo
- `poses_used.yaml`, `data/` → pose e catture della sessione

**Soglie del PASS**: residui media ≤ 1.5 mm e massimo ≤ 3 mm, rotazione media ≤ 0.3°, verifica ≤ 3 mm e 0.6°. Riferimento della board del 21/10: residui di circa 0.85 mm.

Uso nel nodo camera (H2):
```bash
python3 examples/thesis/zivid_target_node.py --calib ~/zivid_ws/calib_board_<data_ora>/T_cam_to_base_Zivid_board.npy ...
```

---

## 4. Verifica finale: touch test (consigliato)

1. Camera: `p_B` del cubetto ID 0 con `zivid_target_node.py --calib <nuova matrice>`.
2. Punte delle dita (pinza aperta) a mano sul centro del marker, robot fermo:
   ```bash
   ros2 topic echo --once --field pose.position /robot1/franka_robot_state_broadcaster/current_pose
   ```
3. Errore = differenza tra le due posizioni. 3 appoggi, in 2–3 punti del tavolo. Atteso circa 2–3 mm.

---

## 5. Quando ricalibrare

- La camera o il robot sono stati spostati o urtati.
- La configurazione del lab è cambiata.
- Il touch test dà più di circa 5 mm.

Cambiare end-effector **non** richiede di ricalibrare: cambia solo il TCP in Desk.

---

## Problemi frequenti

| Sintomo | Causa / soluzione |
|---|---|
| `No joint state / TCP pose received` | driver spento o controller non attivi; controlla i topic `/robot1/...` |
| Molti `robot moved during the capture` | gravity compensation imprecisa (massa in Desk), o mani sul robot durante lo scatto |
| Molti `board not detected` | board non rivolta verso la camera, troppo lontana o con riflessi |
| `Joint state rate too low` | rete o driver lenti: controlla che sul PC del controllo non giri altro |
| `RESULT: FAIL` | guarda le pose con residuo alto: elimina quelle anomale o rifai `record` con pose più varie |

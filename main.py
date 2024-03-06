import mss
import keyboard
import uuid
import time
from PIL import Image
"""""
https://fivesjs.skipser.com/trex-game/
"""""

mon={"top": 441, "left": 688, "width": 250, "height": 200}
sct=mss.mss() #ilgili alanı kesecek ve frame yapacak olan kütüphanemiz
i=0
def record_screen(record_id, key):
    global i
    i+=1
    print("{}:{}".format(key, i)) #key: klavyede bastığımız sey i:kac kez klavyeye bastığımız
    img=sct.grab(mon)
    im=Image.frombytes("RGB", img.size, img.rgb)
    im.save("./img/{}_{}_{}.png".format(key, record_id, i))

is_exit=False
def exit():
    global is_exit
    is_exit=True
keyboard.add_hotkey("esc", lambda : exit())
record_id=uuid.uuid4()
while True:
    if is_exit:
        break
    try:
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(record_id,"up")
            time.sleep(0.1)
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id, "down")
            time.sleep(0.1)
        elif keyboard.is_pressed("right"):
            record_screen(record_id, "right")

            time.sleep(0.1)
    except RuntimeError:
        continue
===============================
I-A. TECH STACK / CÔNG NGHỆ ƯU TIÊN
===============================

Mục tiêu là xây dựng tool automation cho game chạy trên LDPlayer theo hướng:

STABLE
+
FAST
+
RECOVERABLE
+
CONFIG DRIVEN
+
COMPUTER VISION BASED
+
DỄ BÀN GIAO CHO MODEL AI KHÁC

Không tự ý lựa chọn framework hoặc công nghệ mới nếu stack dưới đây đã đáp ứng được yêu cầu.

TECH STACK ƯU TIÊN:

--------------------------------
1. NGÔN NGỮ CHÍNH
--------------------------------

Sử dụng:

Python 3.12+

Python là ngôn ngữ chính cho toàn bộ automation engine.

Lý do:
- thư viện automation phong phú;
- OpenCV hỗ trợ tốt;
- dễ điều khiển ADB;
- dễ xử lý screenshot;
- dễ OCR;
- SQLite tích hợp sẵn;
- dễ tạo GUI;
- dễ debug;
- dễ để model AI khác tiếp tục phát triển.

Không chuyển sang:
- C#;
- Java;
- Node.js;
- Rust;

trừ khi có lý do kỹ thuật rõ ràng và phải ghi vào ARCHITECTURE.md + CHANGELOG.md trước khi thay đổi.

--------------------------------
2. LDPLAYER CONTROL
--------------------------------

Công nghệ chính:

ADB – Android Debug Bridge

Python điều khiển ADB bằng:

subprocess

hoặc wrapper Python nhẹ nếu thật sự cần.

ADB chịu trách nhiệm:

- kiểm tra emulator;
- kết nối LDPlayer;
- kiểm tra device;
- tap;
- swipe;
- nhập text;
- screenshot;
- start game;
- stop game;
- restart game;
- kiểm tra package;
- kiểm tra activity;
- recovery khi game crash.

Các command tương đương cần được wrapper hóa trong:

ADBController

Ví dụ API:

connect()
is_connected()
tap(x, y)
swipe(x1, y1, x2, y2, duration)
input_text(text)
screenshot()
start_app()
stop_app()
restart_app()
get_resolution()

Không gọi subprocess ADB rải rác trong project.

Tất cả phải đi qua ADBController.

--------------------------------
3. LDPLAYER COMMAND LINE
--------------------------------

Có thể sử dụng thêm LDPlayer command line interface nếu môi trường hỗ trợ.

Ví dụ các khả năng có thể dùng:

- start instance;
- stop instance;
- restart instance;
- kiểm tra trạng thái;
- điều chỉnh resolution;
- quản lý emulator.

Tạo:

LDPlayerController

nhưng KHÔNG phụ thuộc toàn bộ automation vào LDPlayer CLI.

Phân chia:

LDPlayerController
→ quản lý emulator

ADBController
→ quản lý Android/game bên trong emulator.

--------------------------------
4. COMPUTER VISION
--------------------------------

Công nghệ chính:

OpenCV

Package:

opencv-python

Dùng OpenCV cho:

- template matching;
- button detection;
- icon detection;
- xác định screen hiện tại;
- NPC detection;
- quest UI detection;
- shop detection;
- popup detection;
- loading screen detection.

Thiết kế:

VisionEngine

API ví dụ:

find_template()
find_all_templates()
template_exists()
wait_for_template()
screen_matches()
get_match_confidence()

Không sử dụng AI/YOLO ngay từ đầu nếu template matching đã giải quyết được.

Ưu tiên:

LEVEL 1:
tọa độ cố định

LEVEL 2:
template matching OpenCV

LEVEL 3:
OCR

LEVEL 4:
object detection/model AI

Chỉ nâng cấp khi level trước không đủ ổn định.

--------------------------------
5. IMAGE PROCESSING
--------------------------------

Sử dụng:

NumPy
Pillow

Mục đích:

- xử lý screenshot;
- crop ROI;
- resize;
- grayscale;
- threshold;
- image normalization;
- chuyển đổi format giữa ADB screenshot và OpenCV.

Không xử lý ảnh nhiều lần không cần thiết.

Nếu có thể:
chụp screenshot một lần rồi chia ROI cho nhiều detector.

--------------------------------
6. SCREENSHOT
--------------------------------

Ưu tiên screenshot trực tiếp qua ADB:

adb exec-out screencap -p

Không ưu tiên screenshot desktop bằng pyautogui vì:

- phụ thuộc vị trí cửa sổ;
- phụ thuộc monitor;
- cửa sổ LDPlayer có thể bị che;
- dễ sai coordinate.

PyAutoGUI chỉ dùng FALLBACK nếu có thao tác bắt buộc nằm ngoài Android screen.

Tạo:

ScreenCapture

API:

capture()
capture_region(x1, y1, x2, y2)
save_debug_image()

--------------------------------
7. OCR
--------------------------------

OCR chỉ sử dụng khi thật sự cần đọc chữ.

Ưu tiên:

PaddleOCR

Fallback:

Tesseract OCR

OCR dùng cho:

- tên nhiệm vụ;
- tên NPC;
- tên vật phẩm;
- số lượng;
- nội dung quest;
- thông báo lỗi;
- text không thể xác định bằng template.

Không OCR toàn màn hình liên tục.

Phải crop ROI trước.

Ví dụ:

Quest text area
↓
crop
↓
preprocess
↓
OCR

Nếu cùng một thông tin có thể xác định bằng:
- icon;
- coordinate;
- template;

thì ưu tiên các cách đó trước OCR.

--------------------------------
8. ANDROID UI AUTOMATION
--------------------------------

Có thể sử dụng:

Appium
+
UiAutomator2

NHƯNG chỉ là công nghệ bổ sung.

Appium dùng khi Android UI có thể truy cập element native như:

- màn hình login;
- popup Android;
- input field;
- dialog hệ thống.

Không coi Appium là automation engine chính của game.

Lý do:

Nhiều game render bằng:

Unity
Cocos
Unreal Engine
OpenGL
SurfaceView

khi đó UiAutomator/Appium có thể không nhìn thấy:

NPC
button
quest
shop
map object.

Trong trường hợp đó phải sử dụng:

ADB + OpenCV.

--------------------------------
9. INPUT / CLICK ENGINE
--------------------------------

Input ưu tiên:

ADB input tap
ADB input swipe

Không dùng PyAutoGUI để click trực tiếp cửa sổ LDPlayer nếu ADB đã làm được.

Tạo:

InputController

API:

tap()
double_tap()
long_press()
swipe()
drag()
input_text()

InputController sử dụng ADBController bên dưới.

--------------------------------
10. CONFIG
--------------------------------

Sử dụng:

YAML

Package:

PyYAML

Các file config ví dụ:

config/
  app.yaml
  accounts.yaml
  maps.yaml
  npcs.yaml
  shops.yaml
  ui_coordinates.yaml
  templates.yaml

Không hard-code:

- account;
- NPC;
- map;
- coordinate;
- item;
- timeout;
- retry;
- resolution;
- master account;

trong source code.

--------------------------------
11. DATA VALIDATION
--------------------------------

Khuyến nghị dùng:

Pydantic

để validate config.

Ví dụ:

AccountConfig
MapConfig
NPCConfig
ShopConfig
AutomationSettings

Nếu config thiếu dữ liệu:

tool phải báo rõ:

CONFIG_ERROR

thay vì chạy và crash ở giữa.

--------------------------------
12. DATABASE
--------------------------------

Sử dụng:

SQLite

thông qua:

sqlite3

hoặc SQLAlchemy nếu project phát triển đủ lớn.

Giai đoạn đầu ưu tiên sqlite3 đơn giản.

SQLite dùng lưu:

accounts
account_status
quest_history
runtime_state
gold_transfer_history
errors
run_history

Không lưu password dạng plaintext trong log.

Nếu password/account được lưu local:
thiết kế để sau này có thể nâng cấp sang Windows Credential Manager hoặc encryption.

--------------------------------
13. CHECKPOINT
--------------------------------

Ngoài SQLite, tạo:

runtime/state.json

cho checkpoint nhanh và dễ debug.

state.json lưu:

current_account
current_state
current_quest
current_map
current_step
last_action
retry_count
timestamp

SQLite dùng lịch sử lâu dài.

state.json dùng runtime recovery.

--------------------------------
14. STATE MACHINE
--------------------------------

Ưu tiên tự xây state machine nhẹ bằng:

Python Enum
+
dataclass

Không cần framework state-machine lớn ngay từ đầu.

Ví dụ:

class BotState(Enum):
    BOOT
    CONNECT_EMULATOR
    LOGIN
    GET_QUEST
    CLASSIFY_QUEST
    NAVIGATE
    INTERACT_NPC
    BUY_ITEM
    CANCEL_QUEST
    RETURN_QUEST
    TRANSFER_GOLD
    LOGOUT
    NEXT_ACCOUNT
    FINISHED

Tạo:

StateMachine

Không điều khiển workflow bằng chuỗi if/else khổng lồ.

--------------------------------
15. RETRY
--------------------------------

Có thể sử dụng:

Tenacity

cho các retry đơn giản.

Ví dụ:

- connect ADB;
- screenshot;
- login;
- click button;
- network retry.

Nhưng retry liên quan workflow/state phải do StateMachine quản lý.

Không tạo nested retry vô hạn.

Mọi retry phải có:

max_attempts
timeout
error code
log
screenshot khi cần.

--------------------------------
16. LOGGING
--------------------------------

Ưu tiên:

Loguru

hoặc Python logging chuẩn.

Nếu dùng Loguru thì thống nhất toàn project.

Log format:

[TIME]
[ACCOUNT]
[STATE]
[ACTION]
[RESULT]

Ví dụ:

[12:15:05] [ACC_03] [NAVIGATE] GO_CITY_02 START
[12:15:11] [ACC_03] [NAVIGATE] GO_CITY_02 OK

Log output:

logs/YYYY-MM-DD.log

Có rotation.

--------------------------------
17. GUI
--------------------------------

GUI khuyến nghị:

PySide6

GUI chỉ là lớp điều khiển.

KHÔNG đưa business logic vào GUI.

Kiến trúc:

PySide6 GUI
↓
BotController
↓
Automation Engine

GUI dự kiến có:

- START
- STOP
- PAUSE
- RESUME
- danh sách account
- trạng thái account
- current account
- current quest
- current map
- current state
- log realtime
- tổng số account completed
- failed
- pending.

Giai đoạn đầu có thể chạy CLI trước.

Không được trì hoãn core automation chỉ để làm GUI đẹp.

--------------------------------
18. MULTITHREADING
--------------------------------

Không chạy nhiều account song song.

Yêu cầu hiện tại:

1 LDPlayer
+
1 account tại một thời điểm.

GUI cần không bị freeze nên có thể dùng:

QThread

hoặc worker thread.

Automation engine:

1 worker chính.

Không tạo nhiều thread thao tác game cùng lúc.

--------------------------------
19. ASYNC
--------------------------------

Không cần asyncio cho core automation ở phiên bản đầu.

Ưu tiên code synchronous dễ debug.

Có thể thêm async sau nếu cần:

- logging;
- monitoring;
- external API.

Không over-engineer.

--------------------------------
20. TEST
--------------------------------

Sử dụng:

pytest

Tạo:

tests/

Unit test cho:

ConfigLoader
QuestClassifier
StateMachine
Checkpoint
MapResolver
NPCResolver

Computer Vision phải có test image.

Ví dụ:

tests/assets/

login_screen.png
quest_npc.png
quest_shop.png
quest_combat.png

Test:

assert vision.detect_login(...)
assert classifier.classify(...) == NPC

--------------------------------
21. MOCK MODE
--------------------------------

Phải có MockADBController.

Mục đích:

Test workflow mà không cần game chạy.

Ví dụ:

MockADBController
MockVisionEngine

cho phép simulate:

LOGIN_OK
QUEST_NPC
NPC_FOUND
QUEST_COMPLETE

Nhờ vậy StateMachine có thể test độc lập.

--------------------------------
22. DRY RUN
--------------------------------

Phải có:

DRY_RUN=true

Khi bật:

vision vẫn chạy

state machine vẫn chạy

log vẫn chạy

NHƯNG các thao tác nguy hiểm sẽ bị block:

BUY ITEM
TRANSFER GOLD
LOGOUT
DELETE/CANCEL nếu được cấu hình.

--------------------------------
23. WATCHDOG
--------------------------------

Xây:

WatchdogService

Không cần công nghệ ngoài quá phức tạp.

Watchdog dựa trên:

- timeout;
- screen detection;
- process state;
- ADB connectivity;
- state duration.

Phát hiện:

GAME_CRASH
GAME_FREEZE
LOADING_TIMEOUT
UNKNOWN_SCREEN
ADB_DISCONNECTED
STATE_TIMEOUT

Recovery:

retry
restart game
restart workflow
skip account

tùy error policy.

--------------------------------
24. ERROR HANDLING
--------------------------------

Tạo enum error code.

Ví dụ:

ADB_001
LOGIN_001
VISION_001
NAV_001
NPC_001
SHOP_001
QUEST_001
TRANSFER_001

Error object chứa:

code
account
state
action
message
screenshot
timestamp
retry_count.

--------------------------------
25. PERFORMANCE
--------------------------------

Không screenshot + OCR toàn màn hình liên tục.

Ưu tiên:

SCREENSHOT
↓
CROP ROI
↓
DETECT

Ví dụ chỉ kiểm tra nút Accept trong vùng:

bottom_right_panel

thay vì toàn màn hình.

Cache template đã load.

Không đọc ảnh template từ disk mỗi vòng loop.

TemplateManager load assets một lần khi startup.

--------------------------------
26. TEMPLATE ASSETS
--------------------------------

Cấu trúc:

assets/

  templates/

    screens/
      login/
      ingame/
      loading/

    quest/
      accept/
      complete/
      cancel/

    npc/

    shop/

    popup/

Không hard-code đường dẫn assets trong nhiều module.

Tạo:

TemplateManager.

--------------------------------
27. TEMPLATE MATCHING
--------------------------------

Template matching phải hỗ trợ:

confidence threshold.

Ví dụ config:

templates:

  quest_accept:
    path: "assets/templates/quest/accept.png"
    confidence: 0.88

Không sử dụng một confidence chung cho mọi template.

Cho phép từng template có threshold riêng.

--------------------------------
28. RESOLUTION
--------------------------------

Resolution chuẩn phải nằm trong config.

Ví dụ:

emulator:
  width: 1280
  height: 720
  dpi: 240

Nếu game chạy resolution khác:

- phát hiện;
- warning;
- hoặc điều chỉnh nếu được hỗ trợ.

Coordinate system phải thống nhất.

--------------------------------
29. COORDINATE NORMALIZATION
--------------------------------

Nếu cần hỗ trợ nhiều resolution:

không hard-code toàn bộ pixel.

Có thể dùng normalized coordinate:

x_ratio = x / base_width
y_ratio = y / base_height

Sau đó convert:

real_x = x_ratio * current_width
real_y = y_ratio * current_height

Nhưng phiên bản đầu ưu tiên một resolution chuẩn để giảm complexity.

--------------------------------
30. PATH FINDING / NAVIGATION
--------------------------------

Không sử dụng machine learning nếu game đã có:

- minimap;
- click-to-move;
- tọa độ;
- map teleport;
- NPC coordinate.

Ưu tiên deterministic navigation.

NavigationEngine quản lý:

go_to_map()
go_to_coordinate()
go_to_npc()
go_to_shop()
return_to_quest_giver()

Nếu game có tọa độ nhân vật đọc được bằng UI/OCR thì có thể bổ sung:

PositionReader.

--------------------------------
31. QUEST CLASSIFIER
--------------------------------

QuestClassifier áp dụng theo thứ tự:

1. Detect icon/template
2. Detect quest state
3. OCR quest text nếu cần
4. keyword/rule based classification

Không cần machine learning ở phiên bản đầu.

Các type:

NPC
SPECIAL_NPC_SHOP
COMBAT
UNKNOWN

UNKNOWN phải không tự động thực hiện action nguy hiểm.

Nếu UNKNOWN:

screenshot
log
skip/cancel tùy config.

--------------------------------
32. PASSWORD / CREDENTIAL
--------------------------------

Không ghi username/password vào log.

Config debug cũng phải mask:

username: user123
password: ******

Có thể dùng YAML trong quá trình development.

Sau này có thể nâng cấp:

Windows Credential Manager

hoặc credential encryption.

Không cần triển khai encryption phức tạp trong Phase 1.

--------------------------------
33. BUILD WINDOWS
--------------------------------

Dùng:

PyInstaller

để đóng gói thành:

AutoQuestTool.exe

Sau khi core hoạt động ổn định.

Không build EXE liên tục trong development.

Development chạy trực tiếp Python.

--------------------------------
34. VERSION CONTROL
--------------------------------

Bắt buộc dùng:

Git

Commit sau từng phase ổn định.

Ví dụ:

phase-01-project-skeleton
phase-02-adb
phase-03-login
phase-04-navigation

HANDOFF.md phải ghi:

last_good_commit

để model khác có thể quay lại version ổn định.

--------------------------------
35. DEVELOPMENT TOOLS
--------------------------------

Khuyến nghị:

Python
Git
VS Code hoặc IDE hiện tại
ADB
LDPlayer
OpenCV
SQLite

Không yêu cầu Docker.

Không yêu cầu server riêng.

Tool phải có thể chạy local trên Windows.

--------------------------------
36. DEPENDENCY GỢI Ý
--------------------------------

requirements.txt dự kiến:

opencv-python
numpy
Pillow
PyYAML
pydantic
loguru
tenacity
pytest
PySide6

OCR chỉ thêm khi Phase OCR bắt đầu:

paddleocr

hoặc:

pytesseract

Appium chỉ thêm khi xác nhận cần:

Appium-Python-Client

Không cài một lượng dependency lớn ngay từ đầu.

--------------------------------
37. KIẾN TRÚC CÔNG NGHỆ
--------------------------------

Kiến trúc mong muốn:

                PySide6 GUI
                     |
                     v
               BotController
                     |
                     v
                StateMachine
                     |
       +-------------+-------------+
       |             |             |
       v             v             v
 AccountManager QuestManager    Watchdog
                     |
       +-------------+-------------+
       |             |             |
       v             v             v
 NavigationEngine NPCInteractor ShopManager
       |             |             |
       +-------------+-------------+
                     |
                     v
                 VisionEngine
               /      |       \
              /       |        \
         OpenCV     OCR       Templates
              \       |        /
               \      |       /
                     v
              InputController
                     |
                     v
                ADBController
                     |
                     v
                  LDPlayer

Runtime persistence:

StateMachine
     |
     +--------> state.json
     |
     +--------> SQLite
     |
     +--------> logs
     |
     +--------> debug screenshots

--------------------------------
38. THỨ TỰ ƯU TIÊN CÔNG NGHỆ
--------------------------------

Khi cần giải quyết một vấn đề UI:

ƯU TIÊN 1:
ADB + coordinate

Nếu chưa đủ:

ƯU TIÊN 2:
ADB + OpenCV template matching

Nếu chưa đủ:

ƯU TIÊN 3:
OpenCV preprocessing + OCR

Nếu chưa đủ:

ƯU TIÊN 4:
Appium/UiAutomator2 nếu element native tồn tại

Nếu vẫn chưa đủ:

ƯU TIÊN 5:
đánh giá object detection như YOLO.

Không nhảy ngay vào AI vision phức tạp.

--------------------------------
39. CÔNG NGHỆ KHÔNG NÊN DÙNG LÀM CORE
--------------------------------

Không dùng làm core:

PyAutoGUI
Selenium
Playwright

Selenium/Playwright dành cho browser, không phù hợp game Android trên LDPlayer.

PyAutoGUI chỉ dùng fallback cho thao tác Windows bên ngoài emulator.

Không xây tool dạng macro record/replay đơn giản.

--------------------------------
40. CÔNG NGHỆ CÓ THỂ BỔ SUNG SAU
--------------------------------

Chỉ xem xét sau khi prototype chứng minh cần:

YOLO
ONNX Runtime
EasyOCR
SQLAlchemy
asyncio
Redis
FastAPI
Docker

Không đưa vào Phase 1 nếu chưa có use case thực tế.

===============================
I-B. QUYẾT ĐỊNH TECH STACK
===============================

Nếu không phát hiện blocker kỹ thuật, stack chính thức là:

LANGUAGE:
Python 3.12+

EMULATOR:
LDPlayer

EMULATOR MANAGEMENT:
LDPlayer CLI

ANDROID CONTROL:
ADB

INPUT:
ADB input

SCREENSHOT:
ADB screencap

VISION:
OpenCV

IMAGE:
NumPy + Pillow

OCR:
PaddleOCR
Tesseract fallback

ANDROID NATIVE UI:
Appium + UiAutomator2 OPTIONAL

CONFIG:
YAML + PyYAML

VALIDATION:
Pydantic

RUNTIME STATE:
JSON

DATABASE:
SQLite

STATE MACHINE:
Python Enum + dataclass

RETRY:
Tenacity

LOG:
Loguru

GUI:
PySide6

TEST:
pytest

PACKAGING:
PyInstaller

VERSION CONTROL:
Git

===============================
I-C. NGUYÊN TẮC PROTOTYPE TRƯỚC
===============================

Trước khi xây toàn bộ hệ thống multi-account, phải chứng minh 5 prototype sau hoạt động.

PROTOTYPE 1 – ADB

Phải chứng minh:

Python
↓
ADB
↓
LDPlayer

có thể:

- screenshot;
- tap;
- swipe;
- input;
- detect resolution.

--------------------------------

PROTOTYPE 2 – COMPUTER VISION

Lấy screenshot thật của game.

Chứng minh OpenCV có thể nhận diện ít nhất:

- một button;
- một màn hình;
- một icon/NPC;

với confidence ổn định.

--------------------------------

PROTOTYPE 3 – LOGIN

Chứng minh:

logout
↓
login account
↓
verify ingame

hoạt động với một account test.

--------------------------------

PROTOTYPE 4 – NAVIGATION

Chứng minh:

current map
↓
go_to_map(city)
↓
go_to_coordinate(x,y)
↓
arrived

hoạt động ổn định.

--------------------------------

PROTOTYPE 5 – QUEST

Chứng minh một quest đơn giản:

GET QUEST
↓
CLASSIFY
↓
GO NPC
↓
INTERACT
↓
RETURN
↓
COMPLETE

Sau khi 5 prototype trên PASS mới xây:

multi-account
gold transfer
watchdog
full GUI.

===============================
I-D. KHÔNG ĐƯỢC OVER-ENGINEER
===============================

Luôn chọn giải pháp đơn giản nhất có thể kiểm chứng.

Ví dụ:

Nếu NPC luôn đứng tại tọa độ cố định:

KHÔNG cần YOLO để tìm NPC trên toàn map.

Chỉ:

go_to_coordinate
↓
crop ROI
↓
OpenCV detect NPC.

Nếu nút luôn xuất hiện tại vùng cố định:

không scan toàn màn hình.

Crop ROI trước.

Nếu quest có icon phân biệt:

không OCR toàn bộ quest description.

Dùng icon trước.

Mục tiêu:

độ ổn định > độ phức tạp.

===============================
I-E. YÊU CẦU MODEL AI PHẢI TUÂN THỦ
===============================

Mọi model tiếp quản project phải:

1. Giữ stack hiện tại nếu đang hoạt động.

2. Không tự đổi:
Python
ADB
OpenCV
SQLite
PySide6

sang framework khác nếu không có blocker.

3. Trước khi thêm dependency:
ghi lý do.

4. Trước khi thêm công nghệ mới:
kiểm tra liệu stack hiện tại có giải quyết được không.

5. Không refactor vì sở thích cá nhân.

6. Ưu tiên tương thích code cũ.

7. Mọi quyết định công nghệ mới phải cập nhật:

ARCHITECTURE.md
CHANGELOG.md
HANDOFF.md

8. Nếu context sắp hết:

KHÔNG bắt đầu migration/refactor lớn.

Hãy ghi:

TECH_DECISION_PENDING

vào HANDOFF.md và để model sau tiếp tục đánh giá.

===============================
I-F. OUTPUT TECHNICAL ĐẦU TIÊN
===============================

Trước khi code PHASE 1, hãy kiểm tra môi trường hiện tại và báo:

PYTHON:
version

ADB:
available / unavailable

LDPLAYER:
detected / not detected

ADB DEVICE:
detected / not detected

RESOLUTION:
width x height

OPENCV:
installed / missing

SQLITE:
available / missing

GIT:
available / missing

Sau đó tạo:

requirements.txt

và chỉ đưa các dependency thật sự cần cho phase hiện tại.

Không cài OCR/Appium/YOLO nếu phase hiện tại chưa dùng tới.

Sau đó tiếp tục PHASE 1 theo PROJECT_SPEC.
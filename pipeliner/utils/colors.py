code_reset = "\33[0m"

code_bold = "\33[1m"
code_italic = "\33[3m"
code_url = "\33[4m"
code_blink = "\33[5m"
code_blink2 = "\33[6m"
code_selected = "\33[7m"

code_black = "\33[30m"
code_red = "\33[31m"
code_green = "\33[32m"
code_yellow = "\33[33m"
code_blue = "\33[34m"
code_violet = "\33[35m"
code_beige = "\33[36m"
code_white = "\33[37m"

code_blackbg = "\33[40m"
code_redbg = "\33[41m"
code_greenbg = "\33[42m"
code_yellowbg = "\33[43m"
code_bluebg = "\33[44m"
code_violetbg = "\33[45m"
code_beigebg = "\33[46m"
code_whitebg = "\33[47m"

code_grey = "\33[90m"
code_red2 = "\33[91m"
code_green2 = "\33[92m"
code_orange = "\33[93m"
code_cyan = "\33[94m"
code_violet2 = "\33[95m"
code_beige2 = "\33[96m"
code_white2 = "\33[97m"

code_greybg = "\33[100m"
code_redbg2 = "\33[101m"
code_greenbg2 = "\33[102m"
code_yellowbg2 = "\33[103m"
code_bluebg2 = "\33[104m"
code_violetbg2 = "\33[105m"
code_beigebg2 = "\33[106m"
code_whitebg2 = "\33[107m"


def red(s):
    return f"{code_red}{s}{code_reset}"


def orange(s):
    return f"{code_yellow}{s}{code_reset}"


def green(s):
    return f"{code_green}{s}{code_reset}"


def yellow(s):
    return f"{code_yellow}{s}{code_reset}"


def blue(s):
    return f"{code_blue}{s}{code_reset}"


def magenta(s):
    return f"{code_violet}{s}{code_reset}"


def cyan(s):
    return f"{code_cyan}{s}{code_reset}"


def orange(s):
    return f"{code_orange}{s}{code_reset}"

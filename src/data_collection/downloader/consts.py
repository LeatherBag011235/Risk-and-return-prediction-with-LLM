import re

#start_pattern = re.compile(r'.*?</ix:resources>\s*</ix:header>.*?', re.IGNORECASE)
start_pattern = re.compile(r'</ix:resources>\s*</ix:header>', re.IGNORECASE)
start_pattern_reserve = re.compile(r'<html>', re.IGNORECASE)
end_pattern = re.compile(r'</html>', re.IGNORECASE)
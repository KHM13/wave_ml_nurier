from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from wave_ml.ml.common.CommonUtil import CommonUtil as util
from datetime import datetime
from multipledispatch import dispatch
import time


class StaticCommonUtil:
    __FASTDATE_TYPE_1 = "%Y-%m-%d %H:%M:%S"
    __FASTDATE_TYPE_2 = "%Y.%m.%d"
    __FASTDATE_TYPE_3 = "%Y%m%d%H%M%S"
    __FASTDATE_TYPE_4 = "%Y%m%d"
    __FASTDATE_TYPE_5 = "%Y"
    __FASTDATE_TYPE_6 = "%H"
    __FASTDATE_TYPE_7 = "%M"
    __FASTDATE_TYPE_8 = "%S"

    FASTDATE_TYPE_yyyy_MM = "%Y.%m"
    INQUIRY_RMS_SVC_C = ["EANBMM98R0", "EANBMM16R0"]

    def getFASTDATE_TYPE_1(self) -> str:
        return self.__FASTDATE_TYPE_1

    def getFASTDATE_TYPE_2(self) -> str:
        return self.__FASTDATE_TYPE_2

    def getFASTDATE_TYPE_3(self) -> str:
        return self.__FASTDATE_TYPE_3

    def getFASTDATE_TYPE_4(self) -> str:
        return self.__FASTDATE_TYPE_4

    def getFASTDATE_TYPE_5(self) -> str:
        return self.__FASTDATE_TYPE_5

    def getFASTDATE_TYPE_6(self) -> str:
        return self.__FASTDATE_TYPE_6

    def getFASTDATE_TYPE_7(self) -> str:
        return self.__FASTDATE_TYPE_7

    def getFASTDATE_TYPE_8(self) -> str:
        return self.__FASTDATE_TYPE_8

    @staticmethod
    def getDateFormatToDate(date_format_type: str, source: str) -> datetime:
        try:
            result = datetime.strptime(source, date_format_type)
        except Exception as e:
            result = datetime
            print(f"[StaticCommonUtil][getDateFormatToDate] ERROR : {e}")
        return result

    @staticmethod
    def getDateFormatToString(date_format_type: str, date: datetime) -> str:
        try:
            return date.strftime(date_format_type)
        except:
            return datetime.strftime(date_format_type)

    def getDateFormatToStringBySTD_GBL_ID(self, date_format_type: str, std_gbl_id: str) -> str:
        try:
            if not self.isNullString(std_gbl_id):
                #tr_dtm = buffer("20")
                tr_dtm = ''.join(["20", std_gbl_id[:12]])

                if len(tr_dtm) == 14:
                    return self.getDateFormatToString(date_format_type, self.getDateFormatToDate(self.__FASTDATE_TYPE_3, tr_dtm))
                else:
                    return self.getDateFormatToString(date_format_type, datetime)
            else:
                return self.getDateFormatToString(date_format_type, datetime)
        except Exception as e:
            print(f"[StaticCommonUtil.getDateFormatToString][Exception]: {e}")
            try:
                return self.getDateFormatToString(date_format_type, datetime)
            except Exception as e:
                print(f"[StaticCommonUtil.getDateFormatToString][Exception]: {e}")
        return ""

    def getTr_dtmFromStd_gbl_id(self, std_gbl_id: str) -> str:
        try:
            if not self.isNullString(std_gbl_id):
                tr_dtm = ''.join(["20", std_gbl_id[:12]])

                if len(tr_dtm) == 14:
                    return self.getDateFormatToString(self.__FASTDATE_TYPE_1, self.getDateFormatToDate(self.__FASTDATE_TYPE_3, tr_dtm))
                else:
                    return self.getDateFormatToString(self.__FASTDATE_TYPE_1, datetime)
            else:
                return self.getDateFormatToString(self.__FASTDATE_TYPE_1, datetime)
        except Exception as e:
            print("\t\t\t[StaticCommonUtil.getTr_dtmFromStd_gbl_id_1][Exception]")
            print(f"{self.__class__} ERROR : {e}")
            logger.error(f"{self.__class__} ERROR : {e}")

            try:
                return self.getDateFormatToString(self.__FASTDATE_TYPE_1, datetime)
            except Exception as e:
                print("\t\t\t[StaticCommonUtil.getTr_dtmFromStd_gbl_id_2][Exception]")
                print(f"{self.__class__} ERROR : {e}")
                logger.error(f"{self.__class__} ERROR : {e}")
        return ""

    def isInquiryEqualsValue(self, value: str) -> bool:
        result = False
        for s in self.INQUIRY_RMS_SVC_C:
            if value.__eq__(s):
                result = True
                break
        return result

    @staticmethod
    def getNanoTime():
        return time.time_ns()

    @staticmethod
    def isNullString(value: str) -> bool:
        result = False
        if value is None or value.strip().__eq__("") or value.strip().__eq__("null"):
            result = True
        return result

    @staticmethod
    def isNumberic(value: str) -> bool:
        result = False
        if value is not None:
            result = value.isnumeric()
        return result

    @staticmethod
    def toString_fromInt(value: int) -> str:
        return str(value)

    @staticmethod
    def toString_fromFloat(value: float) -> str:
        return str(value)

    def toInt_fromString(self, value: str) -> int:
        try:
            if self.isNullString(value):
                return 0
            return int(value)
        except Exception as e:
            print(f"{self.__class__} ERROR : {e}")
            return 0

    @dispatch(float)
    def toInt(self, value: float) -> int:
        if value is not None:
            return int(value)
        return 0

    @dispatch(str)
    def toInt(self, value: str) -> int:
        return int(self.toFloat_fromString(value))

    def toFloat_fromString(self, value: str) -> float:
        if self.isNullString(value):
            value = "0"
        elif not self.isNumberic(value):
            value = "0"
        return float(value)

    @staticmethod
    def get_split(value: str, seperator):
        if value is not None and seperator is not None:
            return value.split(seperator)
        else:
            return None

    @staticmethod
    def get_subString(value: str, start: int, end: int) -> str:
        if value is not None and start < end & len(value) >= end:
            return value[start:end]
        else:
            return ""

    @staticmethod
    def getArrayList_fromStrings(values: list) -> list:
        array = []
        if values and len(values) != 0:
            for s in values:
                array.append(s)
        return array

    @dispatch(list, int, str, int)
    def addList(self, valueList: list, index: int, value: str, maxLength: int):
        result = valueList.copy()
        try:
            if not self.isNullString(value):
                result.remove(value)

                if index >= 0:
                    result.insert(index, value)
                else:
                    result.append(value)

                if maxLength != 0 & result.__sizeof__() > maxLength:
                    result.remove(result.__sizeof__() - 1)
        except Exception as e:
            print(f"{self.__class__} ERROR : {e}")
            return valueList
        return result

    @staticmethod
    @dispatch(list, int, str, int, str, int)
    def addList(valueList: list, index: int, value: str, type1_maxSize: int, type2_char: str, type2_maxSize: int):
        type1_size = 0
        type1_last_index = -1
        type2_size = 0
        type2_last_index = -1

        result = []

        loop_index = 0

        for data in valueList:
            if data.startswith(type2_char):
                type2_size += 1
                type2_last_index = loop_index
            else:
                type1_size += 1
                type1_last_index = loop_index
            result.append(data)
            loop_index += 1

            if loop_index > (type1_maxSize + type2_maxSize):
                break

        if value.startswith(type2_char) & type2_size >= type2_maxSize:
            result.remove(type2_last_index)
        elif type1_size >= type1_maxSize:
            result.remove(type1_last_index)
        result.remove(value)

        if index >= 0:
            result.insert(index, value)
        else:
            result.append(value)

        return result

    @staticmethod
    def check_tel_number(value: str) -> str:
        return value

    @staticmethod
    @dispatch(dict, str)
    def is_init_hashMap(map: dict, string_date: str) -> bool:
        if map is not None:
            if map.get("update") is not None:
                if not string_date.__eq__(map.get("update")):
                    return True
            else:
                return True
        else:
            return True
        return False

    @staticmethod
    @dispatch(dict, int)
    def is_init_hashMap(map: dict, int_date: int) -> bool:
        if map is not None:
            if map.get("update") is not None:
                if map.get("update") != int_date:
                    return True
            else:
                return True
        else:
            return True
        return False

    @staticmethod
    @dispatch(str, dict, str)
    def is_init_hashMap(key_update: str, map: dict, date: str) -> bool:
        if map is not None:
            if map.get(key_update) is not None:
                if not date.__eq__(map.get(key_update)):
                    return True
            else:
                return True
        else:
            return True
        return False

    @staticmethod
    @dispatch(str, dict, int)
    def is_init_hashMap(key_update: str, map: dict, date: int) -> bool:
        if map is not None:
            if map.get(key_update) is not None:
                if map.get(key_update) != date:
                    return True
            else:
                return True
        else:
            return True
        return False

    @staticmethod
    @dispatch(str, dict, float)
    def is_init_hashMap(key_update: str, map: dict, date: float) -> bool:
        if map is not None:
            if map.get(key_update) is not None:
                if map.get(key_update) != date:
                    return True
            else:
                return True
        else:
            return True
        return False

    @dispatch(str, int)
    def get_diff_date(self, date1: str, date2: int) -> int:
        if date2 is None:
            date2 = 0
        return self.get_diff_date(date1, str(date2))

    @dispatch(str, float)
    def get_diff_date(self, date1: str, date2: float) -> int:
        if date2 is None:
            date2 = 0.0
        return self.get_diff_date(date1, str(date2))

    @dispatch(str, str)
    def get_diff_date(self, date1: str, date2: str) -> int:
        try:
            if date2 is None:
                return 0

            date_1 = time.mktime(self.getDateFormatToDate(self.getFASTDATE_TYPE_4(), date1).timetuple())
            date_2 = time.mktime(self.getDateFormatToDate(self.getFASTDATE_TYPE_4(), date2).timetuple())
            return (datetime.fromtimestamp(date_2) - datetime.fromtimestamp(date_1)).days
        except Exception as e:
            print(f"{self.__class__} ERROR : {e}")
        return 0
/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf("#1")
    private static int lineStartIndex2(String s, int start) {
        boolean __cfwr_elem66 
        try {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e96) {
            // ignore
        }
= false;

        if (s.length() == 0) {
            return -1;
        }
        if (start == 0) {
            return 0;
        }
        if (start > s.length()) {
            return -1;
        }
        int newlinePos = s.indexOf("\n", start - 1);
        int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
        int returnPos1 = s.indexOf("\r\n", start - 2);
        int returnPos2 = s.indexOf("\r", start - 1);
        int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
        int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
        int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
        if (lineStart >= s.length()) {
            return -1;
        } else {
            return lineStart;
        }
    }
    private static int __cfwr_util940(Character __cfwr_p0, Character __cfwr_p1, Double __cfwr_p2) {
        return (-489 * -120);
        for (int __cfwr_i99 = 0; __cfwr_i99 < 3; __cfwr_i99++) {
            if (true && true) {
            if ((990L % 40.81) && true) {
            if (true && true) {
            try {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 2; __cfwr_i14++) {
            if (false || (('J' | null) * 'M')) {
            for (int __cfwr_i4 = 0; __cfwr_i4 < 10; __cfwr_i4++) {
            return 'G';
        }
        }
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        }
        }
        }
        }
        while (false) {
            if ((751L / (false - 15.50f)) || true) {
            while (true) {
            try {
            try {
            if (((-786 >> null) ^ -132) && false) {
            return null;
        }
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        } catch (Exception __cfwr_e4) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        if (true || false) {
            try {
            Integer __cfwr_obj86 = null;
        } catch (Exception __cfwr_e64) {
            // ignore
        }
        }
        return 472;
    }
    protected static String __cfwr_temp858(long __cfwr_p0, short __cfwr_p1, Boolean __cfwr_p2) {
        if (((false - 29.01f) << -361) || (true + -93.29)) {
            return null;
        }
        while ((4.78f / false)) {
            if (true || false) {
            while (false) {
            if (((770L | null) & (-685L * null)) || false) {
            Boolean __cfwr_result7 = null;
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        return null;
        Integer __cfwr_var67 = null;
        return "item20";
    }
    static String __cfwr_helper401(Integer __cfwr_p0, String __cfwr_p1, float __cfwr_p2) {
        if (true && false) {
            boolean __cfwr_item79 = true;
        }
        try {
            if (((53 & -96.79f) / (0.77f % -439)) || ('R' >> -702L)) {
            try {
            return null;
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e64) {
            // ignore
        }
        return "data16";
    }
}

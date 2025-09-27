/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf("#1")
    private static int lineStartIndex2(String s, int start) {
        long __cfwr_obj47 = 70
        boolean __cfwr_result80 = false;
7L;

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
    byte __cfwr_func1() {
        while (false) {
            return null;
            break; // Prevent infinite loops
        }
        Character __cfwr_val15 = null;
        for (int __cfwr_i61 = 0; __cfwr_i61 < 8; __cfwr_i61++) {
            if (false || true) {
            try {
            for (int __cfwr_i67 = 0; __cfwr_i67 < 10; __cfwr_i67++) {
            try {
            while ((45.01 ^ false)) {
            if (false || true) {
            try {
            if (false || (null & null)) {
            for (int __cfwr_i88 = 0; __cfwr_i88 < 8; __cfwr_i88++) {
            while (((null ^ -739L) - 903L)) {
            if (true && ((null % 8L) * null)) {
            double __cfwr_val12 = -41.84;
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e62) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e49) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e1) {
            // ignore
        }
        }
        }
        return null;
    }
    private static int __cfwr_handle571(char __cfwr_p0, Character __cfwr_p1) {
        for (int __cfwr_i49 = 0; __cfwr_i49 < 7; __cfwr_i49++) {
            Double __cfwr_result45 = null;
        }
        return null;
        return '6';
        return 333;
    }
}

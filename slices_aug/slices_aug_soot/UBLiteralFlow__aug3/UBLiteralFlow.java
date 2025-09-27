/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf("#1")
    private static int lineStartIndex2(String s, int start) {
        if (((null + 92.94f) & ('J' >> -633L)) || (null & 'g')) {
            while ((-205 << (-145L ^ -54.86))) {
            for (int __cfwr_i5 = 0; __cfwr_i5 < 8; __cfwr_i5++) {
            if (('t' & 32.15f) || true) {
            return ((-19.60 ^ null) + (-21.21 / -123L));
        }
        }
            break; // Prevent infinite loops
        }
        }

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
    private static byte __cfwr_handle977(float __cfwr_p0, double __cfwr_p1, Object __cfwr_p2) {
        try {
            try {
            if (false && (-292 ^ (false % -24.55f))) {
            if (false || (154 / null)) {
            Character __cfwr_temp2 = null;
        }
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        while (false) {
            return false;
            break; // Prevent infinite loops
        }
        return null;
    }
    private Boolean __cfwr_handle614() {
        return false;
        return (117L & 'T');
        return null;
    }
}

/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf("#1")
    private static int lineStartIndex2(String s, int start) {
        while (true) {
            return 87.21f;
            break; // Prevent infinite loops
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
    public static Float __cfwr_util987(Boolean __cfwr_p0, float __cfwr_p1) {
        String __cfwr_val63 = "value7";
        while (false) {
            return -92.64f;
            break; // Prevent infinite loops
        }
        if (false || ((true / true) + (-31.09f & 'V'))) {
            char __cfwr_entry90 = 'v';
        }
        return null;
    }
    public static Long __cfwr_helper809(Double __cfwr_p0) {
        int __cfwr_result40 = -296;
        try {
            return null;
        } catch (Exception __cfwr_e22) {
            // ignore
        }
        for (int __cfwr_i60 = 0; __cfwr_i60 < 3; __cfwr_i60++) {
            if (true && true) {
            return 782;
        }
        }
        for (int __cfwr_i34 = 0; __cfwr_i34 < 4; __cfwr_i34++) {
            return null;
        }
        return null;
    }
}

/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf("#1")
    private static int lineStartIndex2(String s, int start) {
        for (int __cfwr_i49 = 0; __cfwr_i49 < 7; __cfwr_i49++) {
            while (false) {
            try {
            while (true) {
            for (int __cfwr_i10 = 0; __cfwr_i10 < 2; __cfwr_i10++) {
            if (true && ((null >> '9') / 61.21)) {
            for (int __cfwr_i42 = 0; __cfwr_i42 < 5; __cfwr_i42++) {
            try {
            try {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 9; __cfwr_i77++) {
            return -897L;
        }
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        } catch (Exception __cfwr_e31) {
            // ignore
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e75) {
            // ignore
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
    public static Float __cfwr_handle860(Integer __cfwr_p0, char __cfwr_p1) {
        long __cfwr_entry98 = -839L;
        return null;
    }
}

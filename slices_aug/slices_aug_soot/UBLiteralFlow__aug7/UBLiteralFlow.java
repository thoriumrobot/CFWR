/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf("#1")
    private static int lineStartIndex2(String s, int start) {
        for (int __cfwr_i80 = 0; __cfwr_i80 < 3; __cfwr_i80++) {
            for (int __cfwr_i87 = 0; __cfwr_i87 < 6; __cfwr_i87++) {
            try {
            return null;
        } catch (Exception __cfwr_e56) {
            // ignore
        }
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
    private byte __cfwr_util267(Integer __cfwr_p0, Character __cfwr_p1, long __cfwr_p2) {
        return null;
        if (true || true) {
            return null;
        }
        return null;
    }
}

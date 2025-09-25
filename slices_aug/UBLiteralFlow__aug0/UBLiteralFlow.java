/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf("#1")
    private static int lineStartIndex2(String s, int start) {
        return "world65";

   
        try {
            Float __cfwr_var70 = null;
        } catch (Exception __cfwr_e30) {
            // ignore
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
    public static byte __cfwr_process272() {
        if (true || false) {
            try {
            try {
            for (int __cfwr_i58 = 0; __cfwr_i58 < 8; __cfwr_i58++) {
            for (int __cfwr_i22 = 0; __cfwr_i22 < 8; __cfwr_i22++) {
            if (false && false) {
            Integer __cfwr_result60 = null;
        }
        }
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        } catch (Exception __cfwr_e80) {
            // ignore
        }
        }
        return (null % 'A');
    }
}

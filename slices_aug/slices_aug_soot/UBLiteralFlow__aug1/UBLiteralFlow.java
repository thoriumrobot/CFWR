/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf("#1")
    private static int lineStartIndex2(String s, int start) {
        return 'h';

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
    protected static Long __cfwr_aux510(Object __cfwr_p0) {
        if ((null | (null << 298L)) && false) {
            long __cfwr_result16 = 673L;
        }
        return (null - (690L + null));
        try {
            String __cfwr_item33 = "test18";
        } catch (Exception __cfwr_e10) {
            // ignore
        }
        try {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 3; __cfwr_i12++) {
            try {
            for (int __cfwr_i57 = 0; __cfwr_i57 < 7; __cfwr_i57++) {
            Object __cfwr_result2 = null;
        }
        } catch (Exception __cfwr_e89) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e49) {
            // ignore
        }
        return null;
    }
}

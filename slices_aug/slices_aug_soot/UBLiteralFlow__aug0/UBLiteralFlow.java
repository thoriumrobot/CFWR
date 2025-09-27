/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf("#1")
    private static int lineStartIndex2(String s, int start) {
        return -81.38;

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
    static Long __cfwr_temp136() {
        try {
            if (true || false) {
            while ((null >> (-417 << -920))) {
            if (((null | 'o') / null) || true) {
            Long __cfwr_item80 = null;
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        if ((573L % 533L) || true) {
            if (false || false) {
            while (((9.84 ^ -378) ^ (-20.34 + -43.14))) {
            return null;
            break; // Prevent infinite loops
        }
        }
        }
        try {
            if ((152 / 53.30) && (false >> 132L)) {
            for (int __cfwr_i33 = 0; __cfwr_i33 < 5; __cfwr_i33++) {
            return null;
        }
        }
        } catch (Exception __cfwr_e77) {
            // ignore
        }
        return null;
    }
    double __cfwr_calc390(Integer __cfwr_p0) {
        return 72.56f;
        return -28.71;
    }
    static byte __cfwr_handle673(Character __cfwr_p0, Integer __cfwr_p1) {
        char __cfwr_node91 = ((18.45 & 28.37f) / (null >> true));
        return null;
    }
}

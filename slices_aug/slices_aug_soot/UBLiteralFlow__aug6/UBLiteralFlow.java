/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf("#1")
    private static int lineStartIndex2(String s, int start) {
        for (int __cfwr_i12 = 0; __cfwr_i12 < 8; __cfwr_i12++) {
            long __cfwr_obj81 = 464L;
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
    protected static Double __cfwr_proc105() {
        if ((-79.80f | null) && false) {
            for (int __cfwr_i94 = 0; __cfwr_i94 < 3; __cfwr_i94++) {
            try {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 10; __cfwr_i32++) {
            try {
            for (int __cfwr_i54 = 0; __cfwr_i54 < 8; __cfwr_i54++) {
            for (int __cfwr_i60 = 0; __cfwr_i60 < 2; __cfwr_i60++) {
            if ((null + -94.55f) && (true % (null << -510))) {
            for (int __cfwr_i21 = 0; __cfwr_i21 < 6; __cfwr_i21++) {
            if (false || (null >> 373)) {
            for (int __cfwr_i20 = 0; __cfwr_i20 < 4; __cfwr_i20++) {
            if (true || true) {
            Long __cfwr_entry13 = null;
        }
        }
        }
        }
        }
        }
        }
        } catch (Exception __cfwr_e77) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e54) {
            // ignore
        }
        }
        }
        if (true && false) {
            while (((-12.63 - -77.00f) + null)) {
            if (true && ((-8.17f + -568L) & -13.08f)) {
            return null;
        }
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i21 = 0; __cfwr_i21 < 4; __cfwr_i21++) {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 9; __cfwr_i6++) {
            return null;
        }
        }
        return null;
    }
    private static Double __cfwr_func28() {
        if (false || ((false ^ null) - true)) {
            char __cfwr_item96 = 'p';
        }
        if (true && true) {
            if (true && true) {
            while (false) {
            while (true) {
            long __cfwr_result31 = -876L;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        try {
            return -34.19f;
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        try {
            return null;
        } catch (Exception __cfwr_e14) {
            // ignore
        }
        return null;
    }
}

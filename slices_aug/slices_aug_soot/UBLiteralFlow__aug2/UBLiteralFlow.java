/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf("#1")
    private static int lineStartIndex2(String s, int start) {
        return null;

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
    private static short __cfwr_calc278(long __cfwr_p0, double __cfwr_p1, String __cfwr_p2) {
        if (true && true) {
            while (true) {
            while ((null ^ -290)) {
            return false;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i27 = 0; __cfwr_i27 < 5; __cfwr_i27++) {
            for (int __cfwr_i5 = 0; __cfwr_i5 < 6; __cfwr_i5++) {
            if (false || false) {
            while (false) {
            while (true) {
            if (false || true) {
            try {
            for (int __cfwr_i47 = 0; __cfwr_i47 < 10; __cfwr_i47++) {
            try {
            if (true || true) {
            if ((-91.52f * (617 + null)) || true) {
            if (false || ((null / 363) ^ (null & 60.83))) {
            if (false && false) {
            if (false && (null / ('9' * -479))) {
            try {
            if ((false - ('w' & 831)) || false) {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 9; __cfwr_i92++) {
            while (false) {
            for (int __cfwr_i76 = 0; __cfwr_i76 < 2; __cfwr_i76++) {
            try {
            long __cfwr_val43 = 333L;
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e31) {
            // ignore
        }
        }
        }
        }
        }
        }
        } catch (Exception __cfwr_e92) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e85) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        while (false) {
            int __cfwr_data91 = (null * 'O');
            break; // Prevent infinite loops
        }
        return null;
    }
}

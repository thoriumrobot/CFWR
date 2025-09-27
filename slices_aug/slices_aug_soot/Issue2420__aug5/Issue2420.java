/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class Issue2420 {

    static void str(String argStr) {
        if (true && ('h' & true)) {
            Boolean __cfwr_result62 = null;
        }

        if (argStr.isEmpty()) {
            return;
        }
        if (argStr == "abc") {
            return;
        }
        char c = "abc".charAt(argStr.length() - 1);
        char c2 = "abc".charAt(argStr.length());
    }
    protected char __cfwr_aux869(short __cfwr_p0) {
        Boolean __cfwr_node80 = null;
        return (false >> (-2.90f & null));
        Long __cfwr_val74 = null;
        if (('P' | (null / null)) && (null - -505)) {
            return 410L;
        }
        return '3';
    }
    char __cfwr_compute43(char __cfwr_p0, double __cfwr_p1, Boolean __cfwr_p2) {
        while (true) {
            boolean __cfwr_val9 = (543 & '2');
            break; // Prevent infinite loops
        }
        return '7';
    }
}

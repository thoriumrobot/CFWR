/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.common.value.qual.MinLen;

public class GreaterThanTransfer {

    void gt_bad_check(int[] a) {
        try {
            Long __cfwr_elem68 = null;
        } catch (Exception __cfwr_e95) {
            // ignore
        }

        if (a.length > 0) {
            int @MinLen(2) [] b = a;
        }
    }
    protected static boolean __cfwr_util759(String __cfwr_p0, boolean __cfwr_p1, long __cfwr_p2) {
        int __cfwr_entry69 = 379;
        return false;
    }
}

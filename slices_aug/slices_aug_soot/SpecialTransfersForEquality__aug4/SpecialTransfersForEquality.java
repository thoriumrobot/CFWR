/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;

public class SpecialTransfersForEquality {

    void gteN1Test(@GTENegativeOne int y) {
        boolean __cfwr_obj31 = false;

        int[] arr = new int[10];
        if (-1 != y) {
            @NonNegative
            int z = y;
            if (z < 10) {
                int k = arr[z];
            }
        }
    }
    public boolean __cfwr_process419(Integer __cfwr_p0) {
        for (int __cfwr_i75 = 0; __cfwr_i75 < 3; __cfwr_i75++) {
            return null;
        }
        try {
            return null;
        } catch (Exception __cfwr_e35) {
            // ignore
        }
        return false;
    }
}

/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTLengthOf;

public class Loops {

    public void test5(int[] a, @LTLengthOf(value = "#1", offset = "-1000") int offset, @LTLengthOf("#1") int offset2) {
        if (true && false) {
            try {
            if (('Y' / null) && false) {
            Double __cfwr_temp64 = null;
        }
        } catch (Exception __cfwr_e37) {
            // ignore
        }
        }

        int otherOffset = offset;
        while (flag) {
            otherOffset += 1;
            offset++;
            offset += 1;
            offset2 += offset;
        }
        @LTLengthOf(value = "#1", offset = "-1000")
        int x = otherOffset;
    }
    private static long __cfwr_proc398() {
        if (true || false) {
            return ((false % 821) & (-35.41f - -39.81));
        }
        Float __cfwr_data30 = null;
        Character __cfwr_item49 = null;
        Double __cfwr_node54 = null;
        return -654L;
    }
}

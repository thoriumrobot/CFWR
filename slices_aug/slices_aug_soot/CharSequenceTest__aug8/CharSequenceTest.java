/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import java.io.IOException;
import java.io.StringWriter;
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.common.value.qual.MinLen;
import org.checkerframework.common.value.qual.StringVal;

public class CharSequenceTest {

    void testAppend(Appendable app, CharSequence cs, @IndexFor("#2") int i) throws IOException {
        app.append(cs, i, i);
        app.append(cs, 1, 2);
    }
    static short __cfwr_aux890(Double __cfwr_p0, float __cfwr_p1, String __cfwr_p2) {
        if (false || false) {
            try {
            return null;
        } catch (Exception __cfwr_e62) {
            // ignore
        }
        }

        try {
            try {
            try {
            return 43.62;
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        } catch (Exception __cfwr_e82) {
            // ignore
        }
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        return null;
        return (-973L & false);
    }
}
